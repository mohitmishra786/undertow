//! Offline conversion: HF checkpoint (f32/bf16/f16 safetensors, possibly
//! sharded) into a quantized undertow checkpoint laid out for streaming.
//!
//! Output layout:
//! * `experts-{layer:05}.safetensors` — all routed experts of one layer,
//!   so a layer's working set has file locality on disk.
//! * `dense.safetensors` — everything else (attention, shared experts,
//!   embeddings, norms, router), resident at load time.
//! * `model.safetensors.index.json` — standard weight map, so the same
//!   `ShardedModelReader` opens converted and unconverted models alike.
//! * `undertow.json` — conversion manifest (formats, version, source).
//! * `config.json` and tokenizer files copied through untouched.
//!
//! Properties that matter at 600GB scale:
//! * **Constant memory.** Tensors stream through in row chunks; nothing is
//!   ever fully resident. Per-row scales make chunked quantization exact,
//!   not an approximation of whole-tensor quantization.
//! * **Resumable.** Each shard is written atomically (tmp + rename); a
//!   rerun skips shards that already verify and redoes the rest.
//! * **Refuses corrupt input.** Non-finite weights abort with the tensor
//!   name rather than silently quantizing garbage.

use std::collections::BTreeMap;
use std::path::Path;

use undertow_core::{EngineError, QTensor, QuantFormat, Result};
use undertow_io::{PlannedEntry, ShardWriter, ShardedModelReader};

/// Where a tensor belongs, decided by the architecture family.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Disposition {
    /// A routed-expert matrix: quantized to the expert format, stored in
    /// its layer's expert shard.
    Expert { layer: usize },
    /// A dense 2-D matrix (attention, shared experts, embeddings, head):
    /// quantized to the dense format, stored in the dense shard.
    Dense,
    /// Numerically sensitive or non-matrix data (norms, router gate,
    /// biases): stored f32 in the dense shard, never quantized.
    KeepF32,
}

pub type Classifier<'a> = &'a (dyn Fn(&str) -> Disposition + Sync);

#[derive(Debug, Clone, Copy)]
pub struct ConvertOptions {
    pub expert_format: QuantFormat,
    pub dense_format: QuantFormat,
    /// Rows quantized per read chunk. Bounds converter memory at roughly
    /// `row_chunk * in_dim * 4` bytes for the largest tensor.
    pub row_chunk: usize,
    /// Redo shards that already exist and verify.
    pub force: bool,
}

impl Default for ConvertOptions {
    fn default() -> Self {
        Self {
            expert_format: QuantFormat::Int4,
            dense_format: QuantFormat::Int8,
            row_chunk: 1024,
            force: false,
        }
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct ConvertReport {
    pub shards_written: usize,
    pub shards_skipped: usize,
    pub tensors: usize,
    pub bytes_in: u64,
    pub bytes_out: u64,
}

/// Sidecar files copied through when present.
const PASSTHROUGH_FILES: &[&str] = &[
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
];

struct PlannedTensor {
    name: String,
    shape: Vec<usize>,
    target: QuantFormat,
    src_bytes: u64,
}

fn plan_tensor(
    reader: &ShardedModelReader,
    name: &str,
    disposition: Disposition,
    opts: &ConvertOptions,
) -> Result<PlannedTensor> {
    let info = reader.info(name)?;
    // Only 2-D matrices quantize; anything else stays f32 regardless of
    // what the classifier said (a 1-D "expert" tensor would be a bug in
    // the classifier, but storing it f32 is always correct).
    let target = if info.shape.len() == 2 {
        match disposition {
            Disposition::Expert { .. } => opts.expert_format,
            Disposition::Dense => opts.dense_format,
            Disposition::KeepF32 => QuantFormat::F32,
        }
    } else {
        QuantFormat::F32
    };
    Ok(PlannedTensor {
        name: name.to_string(),
        shape: info.shape.clone(),
        target,
        src_bytes: info.nbytes,
    })
}

fn planned_entries(t: &PlannedTensor) -> Vec<PlannedEntry> {
    match t.target {
        QuantFormat::F32 => vec![PlannedEntry {
            name: t.name.clone(),
            dtype: "F32",
            shape: t.shape.clone(),
            nbytes: t.shape.iter().product::<usize>() as u64 * 4,
        }],
        fmt @ (QuantFormat::Int8 | QuantFormat::Int4) => {
            let (o, i) = (t.shape[0], t.shape[1]);
            let payload_shape = if fmt == QuantFormat::Int8 {
                vec![o, i]
            } else {
                vec![o, i.div_ceil(2)]
            };
            vec![
                PlannedEntry {
                    name: t.name.clone(),
                    dtype: "U8",
                    shape: payload_shape,
                    nbytes: fmt.payload_bytes(o, i) as u64,
                },
                PlannedEntry {
                    name: format!("{}.scales", t.name),
                    dtype: "F32",
                    shape: vec![o],
                    nbytes: o as u64 * 4,
                },
            ]
        }
    }
}

/// Stream one tensor through quantization into the shard writer.
fn write_tensor(
    reader: &ShardedModelReader,
    w: &mut ShardWriter,
    t: &PlannedTensor,
    row_chunk: usize,
) -> Result<u64> {
    let mut out_bytes = 0u64;
    if t.target == QuantFormat::F32 || t.shape.len() != 2 {
        // f32 passthrough, still chunked for 2-D tensors.
        if t.shape.len() == 2 {
            let (o, i) = (t.shape[0], t.shape[1]);
            let mut row = 0;
            while row < o {
                let n = (o - row).min(row_chunk);
                let chunk = reader.read_f32_rows(&t.name, row, n)?;
                if let Some(bad) = chunk.data.iter().position(|v| !v.is_finite()) {
                    return Err(non_finite(&t.name, row * i + bad));
                }
                let bytes: Vec<u8> = chunk.data.iter().flat_map(|v| v.to_le_bytes()).collect();
                out_bytes += bytes.len() as u64;
                w.append(&bytes)?;
                row += n;
            }
        } else {
            let full = reader.read_f32(&t.name)?;
            if let Some(bad) = full.data.iter().position(|v| !v.is_finite()) {
                return Err(non_finite(&t.name, bad));
            }
            let bytes: Vec<u8> = full.data.iter().flat_map(|v| v.to_le_bytes()).collect();
            out_bytes += bytes.len() as u64;
            w.append(&bytes)?;
        }
        return Ok(out_bytes);
    }

    // Quantized: per-row scales make row-chunked quantization exactly
    // equal to whole-tensor quantization.
    let (o, i) = (t.shape[0], t.shape[1]);
    let mut scales = Vec::with_capacity(o);
    let mut row = 0;
    while row < o {
        let n = (o - row).min(row_chunk);
        let chunk = reader.read_f32_rows(&t.name, row, n)?;
        let q = QTensor::quantize(&chunk.data, n, i, t.target)
            .map_err(|e| EngineError::Quant(format!("{}: {e}", t.name)))?;
        let (payload, chunk_scales) = q.to_parts();
        out_bytes += payload.len() as u64;
        w.append(&payload)?;
        scales.extend_from_slice(chunk_scales);
        row += n;
    }
    let scale_bytes: Vec<u8> = scales.iter().flat_map(|v| v.to_le_bytes()).collect();
    out_bytes += scale_bytes.len() as u64;
    w.append(&scale_bytes)?;
    Ok(out_bytes)
}

fn non_finite(name: &str, flat: usize) -> EngineError {
    EngineError::Quant(format!(
        "{name}: non-finite value at flat index {flat}; refusing to convert a corrupt tensor"
    ))
}

/// True if `path` is a complete shard containing exactly `names`.
fn shard_verifies(path: &Path, plan: &[PlannedEntry]) -> bool {
    let Ok(reader) = undertow_io::SafetensorsReader::open(path) else {
        return false;
    };
    if reader.tensor_names().count() != plan.len() {
        return false;
    }
    plan.iter().all(|e| {
        reader
            .info(&e.name)
            .map(|info| info.nbytes == e.nbytes && info.shape == e.shape)
            .unwrap_or(false)
    })
}

pub fn convert(
    src: impl AsRef<Path>,
    dst: impl AsRef<Path>,
    classify: Classifier,
    opts: &ConvertOptions,
) -> Result<ConvertReport> {
    let (src, dst) = (src.as_ref(), dst.as_ref());
    if opts.row_chunk == 0 {
        return Err(EngineError::InvalidConfig("row_chunk must be > 0".into()));
    }
    std::fs::create_dir_all(dst)?;
    let reader = ShardedModelReader::open(src)?;

    // Group tensors into output shards, deterministically ordered.
    let mut names: Vec<String> = reader.tensor_names().map(str::to_string).collect();
    names.sort();
    let mut shards: BTreeMap<String, Vec<PlannedTensor>> = BTreeMap::new();
    for name in &names {
        let disposition = classify(name);
        let shard = match disposition {
            Disposition::Expert { layer } => format!("experts-{layer:05}.safetensors"),
            _ => "dense.safetensors".to_string(),
        };
        shards
            .entry(shard)
            .or_default()
            .push(plan_tensor(&reader, name, disposition, opts)?);
    }

    let mut report = ConvertReport::default();
    let mut weight_map = serde_json::Map::new();
    for (shard_name, tensors) in &shards {
        let plan: Vec<PlannedEntry> = tensors.iter().flat_map(planned_entries).collect();
        for e in &plan {
            weight_map.insert(e.name.clone(), serde_json::json!(shard_name));
        }
        let out_path = dst.join(shard_name);
        report.tensors += tensors.len();
        for t in tensors {
            report.bytes_in += t.src_bytes;
        }
        if !opts.force && shard_verifies(&out_path, &plan) {
            report.shards_skipped += 1;
            report.bytes_out += plan.iter().map(|e| e.nbytes).sum::<u64>();
            continue;
        }
        let mut w = ShardWriter::create(&out_path, &plan)?;
        for t in tensors {
            report.bytes_out += write_tensor(&reader, &mut w, t, opts.row_chunk)?;
        }
        w.finish()?;
        report.shards_written += 1;
    }

    // Standard index so ShardedModelReader::open just works on the output.
    let index = serde_json::json!({
        "metadata": { "total_size": report.bytes_out },
        "weight_map": weight_map,
    });
    std::fs::write(
        dst.join("model.safetensors.index.json"),
        serde_json::to_vec_pretty(&index).expect("static json"),
    )?;

    for f in PASSTHROUGH_FILES {
        let s = src.join(f);
        if s.exists() {
            std::fs::copy(&s, dst.join(f))?;
        }
    }

    let manifest = serde_json::json!({
        "format_version": 1,
        "expert_format": opts.expert_format.name(),
        "dense_format": opts.dense_format.name(),
        "source": src.to_string_lossy(),
        "tensors": report.tensors,
        "bytes_out": report.bytes_out,
    });
    std::fs::write(
        dst.join("undertow.json"),
        serde_json::to_vec_pretty(&manifest).expect("static json"),
    )?;

    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use undertow_core::Tensor;
    use undertow_io::write_safetensors;

    fn classify_test(name: &str) -> Disposition {
        if let Some(rest) = name.strip_prefix("layer.") {
            let layer: usize = rest.split('.').next().unwrap().parse().unwrap();
            Disposition::Expert { layer }
        } else if name.starts_with("norm") {
            Disposition::KeepF32
        } else {
            Disposition::Dense
        }
    }

    fn make_src(dir: &Path) {
        let e0 = Tensor::new(
            vec![4, 6],
            (0..24).map(|i| (i as f32 * 0.3).sin()).collect(),
        );
        let e1 = Tensor::new(
            vec![4, 6],
            (0..24).map(|i| (i as f32 * 0.7).cos()).collect(),
        );
        let dense = Tensor::new(
            vec![8, 6],
            (0..48).map(|i| (i as f32 * 0.11).sin()).collect(),
        );
        let norm = Tensor::new(vec![6], (0..6).map(|i| 1.0 + i as f32 * 0.01).collect());
        write_safetensors(
            dir.join("model.safetensors"),
            &[
                ("layer.0.w".into(), &e0),
                ("layer.1.w".into(), &e1),
                ("dense.w".into(), &dense),
                ("norm.w".into(), &norm),
            ],
        )
        .unwrap();
        std::fs::write(dir.join("config.json"), "{\"test\":true}").unwrap();
    }

    #[test]
    fn converts_groups_and_reopens() {
        let src = tempfile::tempdir().unwrap();
        let dst = tempfile::tempdir().unwrap();
        make_src(src.path());
        let opts = ConvertOptions {
            row_chunk: 3, // force chunked path
            ..Default::default()
        };
        let report = convert(src.path(), dst.path(), &classify_test, &opts).unwrap();
        assert_eq!(report.shards_written, 3); // dense + 2 expert layers
        assert_eq!(report.tensors, 4);

        // Output opens through the standard reader and dequantizes close
        // to the source.
        let out = ShardedModelReader::open(dst.path()).unwrap();
        let q = undertow_io::read_qtensor(&out, "layer.0.w", 4, 6).unwrap();
        assert_eq!(q.format(), QuantFormat::Int4);
        let src_reader = ShardedModelReader::open(src.path()).unwrap();
        let orig = src_reader.read_f32("layer.0.w").unwrap();
        let back = q.dequantize();
        for (a, b) in back.iter().zip(&orig.data) {
            assert!((a - b).abs() < 0.15, "{a} vs {b}");
        }
        // Norm stays f32 and exact.
        let norm = out.read_f32("norm.w").unwrap();
        assert_eq!(norm.data.len(), 6);
        assert_eq!(norm.data[3], 1.03);
        // config copied.
        assert!(dst.path().join("config.json").exists());
        assert!(dst.path().join("undertow.json").exists());
    }

    #[test]
    fn chunked_quantization_equals_whole_tensor() {
        let src = tempfile::tempdir().unwrap();
        let a = tempfile::tempdir().unwrap();
        let b = tempfile::tempdir().unwrap();
        make_src(src.path());
        let opts = ConvertOptions {
            row_chunk: 1,
            ..Default::default()
        };
        convert(src.path(), a.path(), &classify_test, &opts).unwrap();
        let opts = ConvertOptions {
            row_chunk: 4096,
            ..Default::default()
        };
        convert(src.path(), b.path(), &classify_test, &opts).unwrap();
        for f in ["dense.safetensors", "experts-00000.safetensors"] {
            let fa = std::fs::read(a.path().join(f)).unwrap();
            let fb = std::fs::read(b.path().join(f)).unwrap();
            assert_eq!(fa, fb, "{f} differs between chunk sizes");
        }
    }

    #[test]
    fn resume_skips_complete_shards() {
        let src = tempfile::tempdir().unwrap();
        let dst = tempfile::tempdir().unwrap();
        make_src(src.path());
        let opts = ConvertOptions::default();
        let first = convert(src.path(), dst.path(), &classify_test, &opts).unwrap();
        assert_eq!(first.shards_written, 3);
        let second = convert(src.path(), dst.path(), &classify_test, &opts).unwrap();
        assert_eq!(second.shards_written, 0);
        assert_eq!(second.shards_skipped, 3);
        // force redoes everything
        let forced = convert(
            src.path(),
            dst.path(),
            &classify_test,
            &ConvertOptions {
                force: true,
                ..opts
            },
        )
        .unwrap();
        assert_eq!(forced.shards_written, 3);
    }

    #[test]
    fn non_finite_source_rejected() {
        let src = tempfile::tempdir().unwrap();
        let dst = tempfile::tempdir().unwrap();
        let bad = Tensor::new(vec![2, 2], vec![1.0, f32::NAN, 0.0, 2.0]);
        write_safetensors(
            src.path().join("model.safetensors"),
            &[("dense.w".into(), &bad)],
        )
        .unwrap();
        let err = convert(
            src.path(),
            dst.path(),
            &classify_test,
            &ConvertOptions::default(),
        )
        .unwrap_err();
        assert!(err.to_string().contains("non-finite"), "{err}");
    }
}
