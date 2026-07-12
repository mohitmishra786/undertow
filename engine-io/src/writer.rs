use std::io::{BufWriter, Write};
use std::path::Path;

use engine_core::{QTensor, Result, Tensor};

/// One tensor to serialize: name, safetensors dtype string, shape, raw
/// little-endian payload.
pub struct TensorEntry {
    pub name: String,
    pub dtype: &'static str,
    pub shape: Vec<usize>,
    pub bytes: Vec<u8>,
}

impl TensorEntry {
    pub fn f32(name: impl Into<String>, shape: Vec<usize>, data: &[f32]) -> Self {
        debug_assert_eq!(shape.iter().product::<usize>(), data.len());
        let mut bytes = Vec::with_capacity(data.len() * 4);
        for v in data {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        Self {
            name: name.into(),
            dtype: "F32",
            shape,
            bytes,
        }
    }

    /// Entries for a [`QTensor`] under `name`: the payload (F32 for
    /// unquantized, U8 packed otherwise) plus a `<name>.scales` F32 tensor
    /// for quantized formats. The payload keeps the logical `[out, in]`
    /// shape for int8 and `[out, ceil(in/2)]` for int4 so shapes stay
    /// truthful about the stored bytes.
    pub fn from_qtensor(name: &str, t: &QTensor) -> Vec<Self> {
        let (payload, scales) = t.to_parts();
        let (o, i) = (t.out_dim(), t.in_dim());
        match t.format() {
            engine_core::QuantFormat::F32 => vec![Self {
                name: name.to_string(),
                dtype: "F32",
                shape: vec![o, i],
                bytes: payload,
            }],
            engine_core::QuantFormat::Int8 => vec![
                Self {
                    name: name.to_string(),
                    dtype: "U8",
                    shape: vec![o, i],
                    bytes: payload,
                },
                Self::f32(format!("{name}.scales"), vec![o], scales),
            ],
            engine_core::QuantFormat::Int4 => vec![
                Self {
                    name: name.to_string(),
                    dtype: "U8",
                    shape: vec![o, i.div_ceil(2)],
                    bytes: payload,
                },
                Self::f32(format!("{name}.scales"), vec![o], scales),
            ],
        }
    }
}

/// Write a `.safetensors` file from prepared entries.
///
/// Written atomically: to `<path>.tmp`, fsynced, then renamed. A crash
/// mid-conversion leaves either the complete old file or a `.tmp` that a
/// resumed conversion overwrites, never a truncated file under the real
/// name.
pub fn write_safetensors_entries(path: impl AsRef<Path>, entries: &[TensorEntry]) -> Result<()> {
    let path = path.as_ref();
    let mut header = serde_json::Map::new();
    let mut offset = 0u64;
    for e in entries {
        let nbytes = e.bytes.len() as u64;
        header.insert(
            e.name.clone(),
            serde_json::json!({
                "dtype": e.dtype,
                "shape": e.shape,
                "data_offsets": [offset, offset + nbytes],
            }),
        );
        offset += nbytes;
    }
    let mut header_bytes = serde_json::to_vec(&serde_json::Value::Object(header))
        .expect("header serialization cannot fail");
    // Spec: pad header with spaces to 8-byte alignment.
    while !header_bytes.len().is_multiple_of(8) {
        header_bytes.push(b' ');
    }

    let tmp = path.with_extension("safetensors.tmp");
    {
        let file = std::fs::File::create(&tmp)?;
        let mut w = BufWriter::new(&file);
        w.write_all(&(header_bytes.len() as u64).to_le_bytes())?;
        w.write_all(&header_bytes)?;
        for e in entries {
            w.write_all(&e.bytes)?;
        }
        w.flush()?;
        file.sync_all()?;
    }
    std::fs::rename(&tmp, path)?;
    Ok(())
}

/// A tensor announced ahead of writing, for the streaming path.
#[derive(Debug, Clone)]
pub struct PlannedEntry {
    pub name: String,
    pub dtype: &'static str,
    pub shape: Vec<usize>,
    pub nbytes: u64,
}

/// Streaming safetensors writer: the header is fixed up front from a plan,
/// payloads are appended in plan order without ever buffering a whole
/// tensor. This is what lets the converter process checkpoints hundreds of
/// times larger than RAM.
///
/// Same atomicity contract as [`write_safetensors_entries`]: data goes to
/// `<path>.tmp` and is renamed into place only after every planned byte
/// arrived and was fsynced. [`ShardWriter::finish`] fails loudly if the
/// byte count does not match the plan.
pub struct ShardWriter {
    file: std::fs::File,
    buf: Vec<u8>,
    tmp: std::path::PathBuf,
    path: std::path::PathBuf,
    expected: u64,
    written: u64,
    finished: bool,
}

impl ShardWriter {
    pub fn create(path: impl AsRef<Path>, plan: &[PlannedEntry]) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let mut header = serde_json::Map::new();
        let mut offset = 0u64;
        for e in plan {
            header.insert(
                e.name.clone(),
                serde_json::json!({
                    "dtype": e.dtype,
                    "shape": e.shape,
                    "data_offsets": [offset, offset + e.nbytes],
                }),
            );
            offset += e.nbytes;
        }
        let mut header_bytes = serde_json::to_vec(&serde_json::Value::Object(header))
            .expect("header serialization cannot fail");
        while !header_bytes.len().is_multiple_of(8) {
            header_bytes.push(b' ');
        }
        let tmp = path.with_extension("safetensors.tmp");
        let file = std::fs::File::create(&tmp)?;
        let mut w = Self {
            file,
            buf: Vec::with_capacity(1 << 20),
            tmp,
            path,
            expected: offset,
            written: 0,
            finished: false,
        };
        w.raw_write(&(header_bytes.len() as u64).to_le_bytes())?;
        w.raw_write(&header_bytes)?;
        Ok(w)
    }

    fn raw_write(&mut self, bytes: &[u8]) -> Result<()> {
        self.buf.extend_from_slice(bytes);
        if self.buf.len() >= 1 << 20 {
            self.flush_buf()?;
        }
        Ok(())
    }

    fn flush_buf(&mut self) -> Result<()> {
        Write::write_all(&mut self.file, &self.buf)?;
        self.buf.clear();
        Ok(())
    }

    /// Append payload bytes (plan order, tensors back to back).
    pub fn append(&mut self, bytes: &[u8]) -> Result<()> {
        self.written += bytes.len() as u64;
        if self.written > self.expected {
            return Err(engine_core::EngineError::Other(format!(
                "{}: writing past planned payload size ({} > {})",
                self.path.display(),
                self.written,
                self.expected
            )));
        }
        self.raw_write(bytes)
    }

    pub fn finish(mut self) -> Result<()> {
        if self.written != self.expected {
            return Err(engine_core::EngineError::Other(format!(
                "{}: payload incomplete ({} of {} bytes)",
                self.path.display(),
                self.written,
                self.expected
            )));
        }
        self.flush_buf()?;
        self.file.sync_all()?;
        std::fs::rename(&self.tmp, &self.path)?;
        self.finished = true;
        Ok(())
    }
}

impl Drop for ShardWriter {
    fn drop(&mut self) {
        if !self.finished {
            // Abandoned mid-write: remove the tmp so a resumed conversion
            // starts clean instead of trusting a truncated file.
            let _ = std::fs::remove_file(&self.tmp);
        }
    }
}

/// Write plain f32 tensors (oracle generator path).
pub fn write_safetensors(path: impl AsRef<Path>, tensors: &[(String, &Tensor)]) -> Result<()> {
    let entries: Vec<TensorEntry> = tensors
        .iter()
        .map(|(n, t)| TensorEntry::f32(n.clone(), t.shape.clone(), &t.data))
        .collect();
    write_safetensors_entries(path, &entries)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{read_qtensor, SafetensorsReader, ShardedModelReader};
    use engine_core::QuantFormat;

    #[test]
    fn roundtrip_f32() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.safetensors");
        let a = Tensor::new(vec![2, 3], vec![1., 2., 3., 4., 5., 6.]);
        let b = Tensor::new(vec![4], vec![-1., 0., 0.5, 2.0]);
        write_safetensors(&path, &[("a".into(), &a), ("b".into(), &b)]).unwrap();

        let r = SafetensorsReader::open(&path).unwrap();
        assert_eq!(r.read_f32("a").unwrap(), a);
        assert_eq!(r.read_f32("b").unwrap(), b);
        assert!(r.read_f32("missing").is_err());
    }

    #[test]
    fn roundtrip_quantized() {
        let dir = tempfile::tempdir().unwrap();
        let (o, i) = (6, 15);
        let w: Vec<f32> = (0..o * i).map(|k| ((k as f32) * 0.13).sin()).collect();
        for fmt in [QuantFormat::F32, QuantFormat::Int8, QuantFormat::Int4] {
            let path = dir.path().join("model.safetensors");
            let t = QTensor::quantize(&w, o, i, fmt).unwrap();
            let entries = TensorEntry::from_qtensor("w", &t);
            write_safetensors_entries(&path, &entries).unwrap();

            let r = ShardedModelReader::open(dir.path()).unwrap();
            let back = read_qtensor(&r, "w", o, i).unwrap();
            assert_eq!(back, t, "{fmt:?} roundtrip");
        }
    }

    #[test]
    fn streaming_writer_roundtrip_and_size_enforcement() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("s.safetensors");
        let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let plan = [PlannedEntry {
            name: "big".into(),
            dtype: "F32",
            shape: vec![6, 4],
            nbytes: 96,
        }];
        let mut w = ShardWriter::create(&path, &plan).unwrap();
        // Append in uneven chunks to exercise buffering.
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        w.append(&bytes[..10]).unwrap();
        w.append(&bytes[10..50]).unwrap();
        w.append(&bytes[50..]).unwrap();
        w.finish().unwrap();
        let r = SafetensorsReader::open(&path).unwrap();
        assert_eq!(r.read_f32("big").unwrap().data, data);

        // Short write must fail at finish and leave no file behind.
        let path2 = dir.path().join("short.safetensors");
        let mut w = ShardWriter::create(&path2, &plan).unwrap();
        w.append(&bytes[..50]).unwrap();
        assert!(w.finish().is_err());
        assert!(!path2.exists());

        // Overlong write must fail at append.
        let mut w = ShardWriter::create(&path2, &plan).unwrap();
        w.append(&bytes).unwrap();
        assert!(w.append(&[0u8]).is_err());
    }

    #[test]
    fn chunked_row_reads_match_full_read() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.safetensors");
        let t = Tensor::new(vec![7, 5], (0..35).map(|i| i as f32 * 0.5).collect());
        write_safetensors(&path, &[("m".into(), &t)]).unwrap();
        let r = SafetensorsReader::open(&path).unwrap();
        let mut assembled = Vec::new();
        let mut row = 0;
        while row < 7 {
            let n = (7 - row).min(3);
            assembled.extend(r.read_f32_rows("m", row, n).unwrap().data);
            row += n;
        }
        assert_eq!(assembled, t.data);
        assert!(r.read_f32_rows("m", 6, 2).is_err(), "OOB rows rejected");
    }

    #[test]
    fn no_tmp_left_behind() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("x.safetensors");
        let a = Tensor::new(vec![1], vec![1.0]);
        write_safetensors(&path, &[("a".into(), &a)]).unwrap();
        assert!(path.exists());
        assert!(!dir.path().join("x.safetensors.tmp").exists());
    }
}
