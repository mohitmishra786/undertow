use std::collections::HashMap;
use std::fs::File;
use std::io;
use std::path::{Path, PathBuf};

#[cfg(unix)]
use std::os::unix::fs::FileExt;

use engine_core::{EngineError, Result, Tensor};

/// Maximum safetensors header we accept (spec allows up to 100MB; real
/// headers are a few MB even for 1T-parameter models).
const MAX_HEADER_LEN: u64 = 100 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
    F32,
    F16,
    BF16,
    U8,
    I8,
}

impl Dtype {
    fn parse(s: &str) -> Result<Self> {
        Ok(match s {
            "F32" => Dtype::F32,
            "F16" => Dtype::F16,
            "BF16" => Dtype::BF16,
            "U8" => Dtype::U8,
            "I8" => Dtype::I8,
            other => {
                return Err(EngineError::Other(format!(
                    "unsupported safetensors dtype: {other}"
                )))
            }
        })
    }

    pub fn byte_size(self) -> usize {
        match self {
            Dtype::F32 => 4,
            Dtype::F16 | Dtype::BF16 => 2,
            Dtype::U8 | Dtype::I8 => 1,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub dtype: Dtype,
    pub shape: Vec<usize>,
    /// Absolute byte offset of the tensor data within the file.
    pub offset: u64,
    pub nbytes: u64,
}

impl TensorInfo {
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }
}

/// One `.safetensors` file, indexed at open. Tensor data is only read on
/// demand via `pread` — opening a 60GB shard costs one header parse.
pub struct SafetensorsReader {
    file: File,
    path: PathBuf,
    tensors: HashMap<String, TensorInfo>,
}

impl SafetensorsReader {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let file = File::open(&path)?;

        let mut len_buf = [0u8; 8];
        read_exact_at(&file, &mut len_buf, 0)?;
        let header_len = u64::from_le_bytes(len_buf);
        if header_len > MAX_HEADER_LEN {
            return Err(EngineError::Other(format!(
                "{}: safetensors header length {header_len} exceeds limit",
                path.display()
            )));
        }

        let mut header = vec![0u8; header_len as usize];
        read_exact_at(&file, &mut header, 8)?;
        let root: serde_json::Value = serde_json::from_slice(&header)
            .map_err(|e| EngineError::Other(format!("{}: bad header json: {e}", path.display())))?;
        let obj = root.as_object().ok_or_else(|| {
            EngineError::Other(format!("{}: header is not an object", path.display()))
        })?;

        let data_start = 8 + header_len;
        let file_len = file.metadata()?.len();
        let mut tensors = HashMap::new();
        for (name, v) in obj {
            if name == "__metadata__" {
                continue;
            }
            let bad = || EngineError::Other(format!("{}: bad entry for {name}", path.display()));
            let dtype = Dtype::parse(v.get("dtype").and_then(|d| d.as_str()).ok_or_else(bad)?)?;
            let shape: Vec<usize> = v
                .get("shape")
                .and_then(|s| s.as_array())
                .ok_or_else(bad)?
                .iter()
                .map(|x| x.as_u64().map(|u| u as usize))
                .collect::<Option<_>>()
                .ok_or_else(bad)?;
            let offs = v
                .get("data_offsets")
                .and_then(|o| o.as_array())
                .ok_or_else(bad)?;
            let (start, end) = match (offs[0].as_u64(), offs[1].as_u64()) {
                (Some(s), Some(e)) if e >= s => (s, e),
                _ => return Err(bad()),
            };
            let nbytes = end - start;
            // Untrusted header: sizes must be self-consistent and in-bounds
            // before any allocation or read happens downstream.
            let expected = shape.iter().product::<usize>() as u64 * dtype.byte_size() as u64;
            if nbytes != expected || data_start + end > file_len {
                return Err(EngineError::Other(format!(
                    "{}: tensor {name} offsets inconsistent with shape/dtype/file size",
                    path.display()
                )));
            }
            tensors.insert(
                name.clone(),
                TensorInfo {
                    dtype,
                    shape,
                    offset: data_start + start,
                    nbytes,
                },
            );
        }

        Ok(Self {
            file,
            path,
            tensors,
        })
    }

    pub fn tensor_names(&self) -> impl Iterator<Item = &str> {
        self.tensors.keys().map(String::as_str)
    }

    pub fn has(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    pub fn info(&self, name: &str) -> Result<&TensorInfo> {
        self.tensors
            .get(name)
            .ok_or_else(|| EngineError::TensorNotFound(name.to_string()))
    }

    /// Read raw bytes of a tensor (future quantized payloads).
    pub fn read_raw(&self, name: &str) -> Result<Vec<u8>> {
        let info = self.info(name)?;
        let mut buf = vec![0u8; info.nbytes as usize];
        read_exact_at(&self.file, &mut buf, info.offset)?;
        Ok(buf)
    }

    /// Read rows `[row_start, row_start + nrows)` of a 2-D tensor as f32.
    /// This is the constant-memory path the converter streams through:
    /// a 10GB embedding matrix is processed in row chunks without ever
    /// being resident at once.
    pub fn read_f32_rows(&self, name: &str, row_start: usize, nrows: usize) -> Result<Tensor> {
        let info = self.info(name)?.clone();
        if info.shape.len() != 2 {
            return Err(EngineError::Other(format!(
                "{name}: read_f32_rows requires a 2-D tensor, shape is {:?}",
                info.shape
            )));
        }
        let (out_dim, in_dim) = (info.shape[0], info.shape[1]);
        if row_start + nrows > out_dim {
            return Err(EngineError::Other(format!(
                "{name}: rows {row_start}..{} out of bounds for {out_dim}",
                row_start + nrows
            )));
        }
        let elem = info.dtype.byte_size();
        let offset = info.offset + (row_start * in_dim * elem) as u64;
        let mut raw = vec![0u8; nrows * in_dim * elem];
        read_exact_at(&self.file, &mut raw, offset)?;
        let data = decode_f32(&raw, info.dtype, name, &self.path)?;
        Ok(Tensor::new(vec![nrows, in_dim], data))
    }

    /// Read a tensor and convert to f32 (from F32, F16 or BF16 storage).
    pub fn read_f32(&self, name: &str) -> Result<Tensor> {
        let info = self.info(name)?.clone();
        let raw = self.read_raw(name)?;
        let data = decode_f32(&raw, info.dtype, name, &self.path)?;
        Ok(Tensor::new(info.shape, data))
    }
}

fn decode_f32(raw: &[u8], dtype: Dtype, name: &str, path: &Path) -> Result<Vec<f32>> {
    Ok(match dtype {
        Dtype::F32 => raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        Dtype::BF16 => raw
            .chunks_exact(2)
            .map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect(),
        Dtype::F16 => raw
            .chunks_exact(2)
            .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect(),
        other => {
            return Err(EngineError::Other(format!(
                "{}: cannot convert {other:?} tensor {name} to f32",
                path.display()
            )))
        }
    })
}

/// A model directory: either a single `model.safetensors` or a sharded
/// checkpoint described by `model.safetensors.index.json`.
pub struct ShardedModelReader {
    /// Shard readers, opened lazily-once and kept (header only, cheap).
    shards: Vec<SafetensorsReader>,
    /// tensor name -> index into `shards`.
    index: HashMap<String, usize>,
}

impl ShardedModelReader {
    pub fn open(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref();
        let index_path = dir.join("model.safetensors.index.json");
        if index_path.exists() {
            let root: serde_json::Value = serde_json::from_slice(&std::fs::read(&index_path)?)
                .map_err(|e| {
                    EngineError::Other(format!("{}: bad index json: {e}", index_path.display()))
                })?;
            let weight_map = root
                .get("weight_map")
                .and_then(|m| m.as_object())
                .ok_or_else(|| {
                    EngineError::Other(format!("{}: missing weight_map", index_path.display()))
                })?;
            let mut shards = Vec::new();
            let mut shard_ids: HashMap<String, usize> = HashMap::new();
            let mut index = HashMap::new();
            for (tensor, file) in weight_map {
                let file = file.as_str().ok_or_else(|| {
                    EngineError::Other(format!("{}: bad weight_map entry", index_path.display()))
                })?;
                let id = match shard_ids.get(file) {
                    Some(&id) => id,
                    None => {
                        let id = shards.len();
                        shards.push(SafetensorsReader::open(dir.join(file))?);
                        shard_ids.insert(file.to_string(), id);
                        id
                    }
                };
                index.insert(tensor.clone(), id);
            }
            Ok(Self { shards, index })
        } else {
            let single = SafetensorsReader::open(dir.join("model.safetensors"))?;
            let index = single
                .tensor_names()
                .map(|n| (n.to_string(), 0usize))
                .collect();
            Ok(Self {
                shards: vec![single],
                index,
            })
        }
    }

    fn shard_for(&self, name: &str) -> Result<&SafetensorsReader> {
        self.index
            .get(name)
            .map(|&i| &self.shards[i])
            .ok_or_else(|| EngineError::TensorNotFound(name.to_string()))
    }

    pub fn has(&self, name: &str) -> bool {
        self.index.contains_key(name)
    }

    pub fn tensor_names(&self) -> impl Iterator<Item = &str> {
        self.index.keys().map(String::as_str)
    }

    pub fn info(&self, name: &str) -> Result<&TensorInfo> {
        self.shard_for(name)?.info(name)
    }

    pub fn read_f32(&self, name: &str) -> Result<Tensor> {
        self.shard_for(name)?.read_f32(name)
    }

    pub fn read_f32_rows(&self, name: &str, row_start: usize, nrows: usize) -> Result<Tensor> {
        self.shard_for(name)?.read_f32_rows(name, row_start, nrows)
    }

    pub fn read_raw(&self, name: &str) -> Result<Vec<u8>> {
        self.shard_for(name)?.read_raw(name)
    }
}

/// Read a weight matrix as a [`QTensor`], transparently handling both
/// plain checkpoints (F32/BF16/F16 tensor) and converted quantized ones
/// (`<name>` U8 payload + `<name>.scales` F32).
///
/// The quantized format is deduced from the payload size against the
/// logical dims: `out*in` bytes is int8, `out*ceil(in/2)` is int4. The two
/// only collide at `in_dim == 1`, which no real projection has; int8 wins
/// there.
pub fn read_qtensor(
    reader: &ShardedModelReader,
    name: &str,
    out_dim: usize,
    in_dim: usize,
) -> Result<engine_core::QTensor> {
    use engine_core::{QTensor, QuantFormat};
    let scales_name = format!("{name}.scales");
    if reader.has(&scales_name) {
        let payload = reader.read_raw(name)?;
        let scales_t = reader.read_f32(&scales_name)?;
        if scales_t.shape != [out_dim] {
            return Err(EngineError::ShapeMismatch {
                name: scales_name,
                expected: vec![out_dim],
                got: scales_t.shape,
            });
        }
        let fmt = if payload.len() == QuantFormat::Int8.payload_bytes(out_dim, in_dim) {
            QuantFormat::Int8
        } else if payload.len() == QuantFormat::Int4.payload_bytes(out_dim, in_dim) {
            QuantFormat::Int4
        } else {
            return Err(EngineError::Other(format!(
                "{name}: payload of {} bytes matches neither int8 nor int4 for [{out_dim}, {in_dim}]",
                payload.len()
            )));
        };
        QTensor::from_quantized(fmt, payload, scales_t.data, out_dim, in_dim)
            .map_err(EngineError::Quant)
    } else {
        let t = reader.read_f32(name)?;
        if t.shape != [out_dim, in_dim] {
            return Err(EngineError::ShapeMismatch {
                name: name.to_string(),
                expected: vec![out_dim, in_dim],
                got: t.shape,
            });
        }
        Ok(QTensor::from_f32(t.data, out_dim, in_dim))
    }
}

#[cfg(unix)]
fn read_exact_at(file: &File, buf: &mut [u8], offset: u64) -> io::Result<()> {
    file.read_exact_at(buf, offset)
}

#[cfg(not(unix))]
fn read_exact_at(file: &File, buf: &mut [u8], offset: u64) -> io::Result<()> {
    // Windows fallback: seek_read equivalent via FileExt.
    use std::os::windows::fs::FileExt as WinFileExt;
    let mut done = 0;
    while done < buf.len() {
        let n = file.seek_read(&mut buf[done..], offset + done as u64)?;
        if n == 0 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "eof in read_exact_at",
            ));
        }
        done += n;
    }
    Ok(())
}
