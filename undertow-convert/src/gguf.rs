use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek};
use std::path::{Path, PathBuf};

#[cfg(unix)]
use std::os::unix::fs::FileExt;

use undertow_core::{EngineError, Result, Tensor};
use undertow_io::{Dtype, TensorInfo};

use crate::{
    convert_from_source, Classifier, ConvertOptions, ConvertReport, Disposition, TensorSource,
};

const GGUF_MAGIC: u32 = 0x46554747; // "GGUF" in little-endian

#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgmlDtype {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
    BF16 = 30,
}

impl GgmlDtype {
    pub fn from_u32(val: u32) -> Result<Self> {
        match val {
            0 => Ok(Self::F32),
            1 => Ok(Self::F16),
            2 => Ok(Self::Q4_0),
            3 => Ok(Self::Q4_1),
            6 => Ok(Self::Q5_0),
            7 => Ok(Self::Q5_1),
            8 => Ok(Self::Q8_0),
            9 => Ok(Self::Q8_1),
            12 => Ok(Self::Q4_K),
            13 => Ok(Self::Q5_K),
            14 => Ok(Self::Q6_K),
            15 => Ok(Self::Q8_K),
            30 => Ok(Self::BF16),
            other => Err(EngineError::Other(format!(
                "unsupported GGUF tensor type {other}"
            ))),
        }
    }

    pub fn block_size(self) -> usize {
        match self {
            Self::F32 | Self::F16 | Self::BF16 => 1,
            Self::Q4_0 | Self::Q4_1 | Self::Q5_0 | Self::Q5_1 | Self::Q8_0 | Self::Q8_1 => 32,
            Self::Q4_K | Self::Q5_K | Self::Q6_K | Self::Q8_K => 256,
        }
    }

    pub fn type_size(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::BF16 => 2,
            Self::Q4_0 => 18,
            Self::Q4_1 => 20,
            Self::Q5_0 => 22,
            Self::Q5_1 => 24,
            Self::Q8_0 => 34,
            Self::Q8_1 => 36,
            Self::Q4_K => 144,
            Self::Q5_K => 176,
            Self::Q6_K => 210,
            Self::Q8_K => 292,
        }
    }
}

pub fn dequantize_blocks(dtype: GgmlDtype, raw: &[u8], out: &mut [f32]) -> Result<()> {
    let block_size = dtype.block_size();
    let type_size = dtype.type_size();
    let num_blocks = out.len() / block_size;
    if raw.len() < num_blocks * type_size {
        return Err(EngineError::Other(format!(
            "short raw buffer for dequantization: got {} bytes, needed {}",
            raw.len(),
            num_blocks * type_size
        )));
    }

    match dtype {
        GgmlDtype::F32 => {
            for (i, chunk) in raw.chunks_exact(4).take(out.len()).enumerate() {
                out[i] = f32::from_le_bytes(chunk.try_into().unwrap());
            }
        }
        GgmlDtype::F16 => {
            for (i, chunk) in raw.chunks_exact(2).take(out.len()).enumerate() {
                out[i] = half::f16::from_le_bytes(chunk.try_into().unwrap()).to_f32();
            }
        }
        GgmlDtype::BF16 => {
            for (i, chunk) in raw.chunks_exact(2).take(out.len()).enumerate() {
                out[i] = half::bf16::from_le_bytes(chunk.try_into().unwrap()).to_f32();
            }
        }
        GgmlDtype::Q8_0 => {
            for (b_idx, chunk) in raw.chunks_exact(34).take(num_blocks).enumerate() {
                let d = half::f16::from_le_bytes([chunk[0], chunk[1]]).to_f32();
                let qs = &chunk[2..34];
                let out_slice = &mut out[b_idx * 32..(b_idx + 1) * 32];
                for (j, &q) in qs.iter().enumerate() {
                    out_slice[j] = (q as i8 as f32) * d;
                }
            }
        }
        GgmlDtype::Q4_0 => {
            for (b_idx, chunk) in raw.chunks_exact(18).take(num_blocks).enumerate() {
                let d = half::f16::from_le_bytes([chunk[0], chunk[1]]).to_f32();
                let qs = &chunk[2..18];
                let out_slice = &mut out[b_idx * 32..(b_idx + 1) * 32];
                for j in 0..16 {
                    let v0 = (qs[j] & 0x0F) as i8 - 8;
                    let v1 = (qs[j] >> 4) as i8 - 8;
                    out_slice[j] = (v0 as f32) * d;
                    out_slice[j + 16] = (v1 as f32) * d;
                }
            }
        }
        GgmlDtype::Q4_K => {
            fn get_scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
                if j < 4 {
                    (q[j] & 63, q[j + 4] & 63)
                } else {
                    (
                        (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4),
                        (q[j + 4] >> 4) | ((q[j] >> 6) << 4),
                    )
                }
            }

            for (b_idx, chunk) in raw.chunks_exact(144).take(num_blocks).enumerate() {
                let d = half::f16::from_le_bytes([chunk[0], chunk[1]]).to_f32();
                let min = half::f16::from_le_bytes([chunk[2], chunk[3]]).to_f32();
                let scales = &chunk[4..16];
                let qs = &chunk[16..144];

                let mut is = 0;
                let mut q_idx = 0;
                let mut y_idx = b_idx * 256;

                for _ in (0..256).step_by(64) {
                    let (sc1, m1) = get_scale_min_k4(is, scales);
                    let d1 = d * sc1 as f32;
                    let m1 = min * m1 as f32;

                    let (sc2, m2) = get_scale_min_k4(is + 1, scales);
                    let d2 = d * sc2 as f32;
                    let m2 = min * m2 as f32;

                    for l in 0..32 {
                        out[y_idx + l] = d1 * (qs[q_idx + l] & 0xF) as f32 - m1;
                        out[y_idx + 32 + l] = d2 * (qs[q_idx + l] >> 4) as f32 - m2;
                    }
                    q_idx += 32;
                    y_idx += 64;
                    is += 2;
                }
            }
        }
        GgmlDtype::Q6_K => {
            for (b_idx, chunk) in raw.chunks_exact(210).take(num_blocks).enumerate() {
                let d = half::f16::from_le_bytes([chunk[208], chunk[209]]).to_f32();
                let ql = &chunk[0..128];
                let qh = &chunk[128..192];
                let sc = &chunk[192..208];

                let mut ql_idx = 0;
                let mut qh_idx = 0;
                let mut sc_idx = 0;
                let mut y_idx = b_idx * 256;

                for _ in (0..256).step_by(128) {
                    for l in 0..32 {
                        let is = l / 16;
                        let q1 =
                            (((ql[ql_idx + l] & 0xF) | ((qh[qh_idx + l] & 3) << 4)) as i8) - 32;
                        let q2 = (((ql[ql_idx + l + 32] & 0xF) | (((qh[qh_idx + l] >> 2) & 3) << 4))
                            as i8)
                            - 32;
                        let q3 = (((ql[ql_idx + l] >> 4) | (((qh[qh_idx + l] >> 4) & 3) << 4))
                            as i8)
                            - 32;
                        let q4 = (((ql[ql_idx + l + 32] >> 4) | (((qh[qh_idx + l] >> 6) & 3) << 4))
                            as i8)
                            - 32;

                        out[y_idx + l] = d * (sc[sc_idx + is] as i8 as f32) * (q1 as f32);
                        out[y_idx + l + 32] = d * (sc[sc_idx + is + 2] as i8 as f32) * (q2 as f32);
                        out[y_idx + l + 64] = d * (sc[sc_idx + is + 4] as i8 as f32) * (q3 as f32);
                        out[y_idx + l + 96] = d * (sc[sc_idx + is + 6] as i8 as f32) * (q4 as f32);
                    }
                    y_idx += 128;
                    ql_idx += 64;
                    qh_idx += 32;
                    sc_idx += 8;
                }
            }
        }
        _ => {
            return Err(EngineError::Other(format!(
                "unsupported dequantization dtype {dtype:?}"
            )))
        }
    }

    Ok(())
}

#[derive(Debug, Clone)]
pub enum GgufMetadataValue {
    Uint8(u8),
    Int8(i8),
    Uint16(u16),
    Int16(i16),
    Uint32(u32),
    Int32(i32),
    Float32(f32),
    Bool(bool),
    String(String),
    Array(Vec<GgufMetadataValue>),
    Uint64(u64),
    Int64(i64),
    Float64(f64),
}

impl GgufMetadataValue {
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Self::Uint8(v) => Some(*v as u64),
            Self::Uint16(v) => Some(*v as u64),
            Self::Uint32(v) => Some(*v as u64),
            Self::Uint64(v) => Some(*v),
            Self::Int8(v) if *v >= 0 => Some(*v as u64),
            Self::Int16(v) if *v >= 0 => Some(*v as u64),
            Self::Int32(v) if *v >= 0 => Some(*v as u64),
            Self::Int64(v) if *v >= 0 => Some(*v as u64),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Self::Float32(v) => Some(*v as f64),
            Self::Float64(v) => Some(*v),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct GgufTensorEntry {
    pub name: String,
    pub shape: Vec<usize>,
    pub file_offset: u64,
    pub dtype: GgmlDtype,
    pub bytes_per_row: usize,
    pub in_dim: usize,
    pub out_dim: usize,
}

pub struct GgufReader {
    file: File,
    #[allow(dead_code)]
    path: PathBuf,
    tensors: HashMap<String, GgufTensorEntry>,
    pub metadata: HashMap<String, GgufMetadataValue>,
}

impl GgufReader {
    pub fn path(&self) -> &Path {
        &self.path
    }
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let mut file = File::open(&path)?;

        let mut magic_buf = [0u8; 4];
        file.read_exact(&mut magic_buf)?;
        let magic = u32::from_le_bytes(magic_buf);
        if magic != GGUF_MAGIC {
            return Err(EngineError::Other(format!(
                "{}: not a valid GGUF file (magic {:08x})",
                path.display(),
                magic
            )));
        }

        let mut buf4 = [0u8; 4];
        file.read_exact(&mut buf4)?;
        let version = u32::from_le_bytes(buf4);
        if version != 2 && version != 3 {
            return Err(EngineError::Other(format!(
                "{}: unsupported GGUF version {}",
                path.display(),
                version
            )));
        }

        let mut buf8 = [0u8; 8];
        file.read_exact(&mut buf8)?;
        let tensor_count = u64::from_le_bytes(buf8);
        file.read_exact(&mut buf8)?;
        let metadata_kv_count = u64::from_le_bytes(buf8);

        let mut metadata = HashMap::new();
        for _ in 0..metadata_kv_count {
            let key = read_string(&mut file)?;
            let val = read_metadata_value(&mut file)?;
            metadata.insert(key, val);
        }

        let alignment = metadata
            .get("general.alignment")
            .and_then(|v| v.as_u64())
            .unwrap_or(32);

        let arch = metadata
            .get("general.architecture")
            .and_then(|v| v.as_str())
            .unwrap_or("deepseek2")
            .to_string();

        let mut raw_tensors = Vec::new();
        for _ in 0..tensor_count {
            let name = read_string(&mut file)?;
            file.read_exact(&mut buf4)?;
            let n_dims = u32::from_le_bytes(buf4) as usize;
            let mut dims = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                file.read_exact(&mut buf8)?;
                dims.push(u64::from_le_bytes(buf8));
            }
            file.read_exact(&mut buf4)?;
            let raw_type = u32::from_le_bytes(buf4);
            let dtype = GgmlDtype::from_u32(raw_type)?;
            file.read_exact(&mut buf8)?;
            let offset = u64::from_le_bytes(buf8);

            raw_tensors.push((name, dims, dtype, offset));
        }

        let current_offset = file.stream_position()?;
        let data_start = current_offset.div_ceil(alignment) * alignment;

        let mut tensors = HashMap::new();
        for (raw_name, dims, dtype, offset) in raw_tensors {
            let abs_offset = data_start + offset;
            let n_dims = dims.len();

            if n_dims == 3 {
                // 3D expert tensor: dims = [in_dim, out_dim, num_experts] in GGUF
                let in_dim = dims[0] as usize;
                let out_dim = dims[1] as usize;
                let num_experts = dims[2] as usize;
                let block_size = dtype.block_size();
                let type_size = dtype.type_size();
                let bytes_per_row = (in_dim / block_size) * type_size;
                let bytes_per_expert = out_dim * bytes_per_row;

                for e in 0..num_experts {
                    let mapped_name = map_expert_tensor_name(&raw_name, e, &arch);
                    let file_offset = abs_offset + (e * bytes_per_expert) as u64;
                    tensors.insert(
                        mapped_name.clone(),
                        GgufTensorEntry {
                            name: mapped_name,
                            shape: vec![out_dim, in_dim],
                            file_offset,
                            dtype,
                            bytes_per_row,
                            in_dim,
                            out_dim,
                        },
                    );
                }
            } else if n_dims == 2 {
                // 2D matrix: dims = [in_dim, out_dim] in GGUF
                let in_dim = dims[0] as usize;
                let out_dim = dims[1] as usize;
                let block_size = dtype.block_size();
                let type_size = dtype.type_size();
                let bytes_per_row = (in_dim / block_size) * type_size;
                let mapped_name = map_standard_tensor_name(&raw_name, &arch);

                tensors.insert(
                    mapped_name.clone(),
                    GgufTensorEntry {
                        name: mapped_name,
                        shape: vec![out_dim, in_dim],
                        file_offset: abs_offset,
                        dtype,
                        bytes_per_row,
                        in_dim,
                        out_dim,
                    },
                );
            } else if n_dims == 1 {
                // 1D tensor: dims = [len]
                let len = dims[0] as usize;
                let block_size = dtype.block_size();
                let type_size = dtype.type_size();
                let bytes_per_row = (len / block_size) * type_size;
                let mapped_name = map_standard_tensor_name(&raw_name, &arch);

                tensors.insert(
                    mapped_name.clone(),
                    GgufTensorEntry {
                        name: mapped_name,
                        shape: vec![len],
                        file_offset: abs_offset,
                        dtype,
                        bytes_per_row,
                        in_dim: len,
                        out_dim: 1,
                    },
                );
            }
        }

        Ok(Self {
            file,
            path,
            tensors,
            metadata,
        })
    }

    /// Synthesize an Undertow/HuggingFace-compatible `config.json` from GGUF metadata.
    pub fn synthesize_config(&self) -> Result<serde_json::Value> {
        let arch = self
            .metadata
            .get("general.architecture")
            .and_then(|v| v.as_str())
            .unwrap_or("deepseek2");

        let model_type = match arch {
            "deepseek2" => "deepseek_v2",
            "deepseek" | "deepseek3" => "deepseek_v3",
            "mixtral" => "mixtral",
            "qwen2_moe" | "qwen3_moe" => "qwen2_moe",
            other => other,
        };

        let get_u = |suffix: &str| -> Option<u64> {
            self.metadata
                .get(&format!("{arch}.{suffix}"))
                .and_then(|v| v.as_u64())
        };
        let get_f = |suffix: &str| -> Option<f64> {
            self.metadata
                .get(&format!("{arch}.{suffix}"))
                .and_then(|v| v.as_f64())
        };

        let hidden_size = get_u("embedding_length").unwrap_or(2048);
        let num_hidden_layers = get_u("block_count").unwrap_or(28);
        let num_attention_heads = get_u("attention.head_count").unwrap_or(16);
        let num_key_value_heads = get_u("attention.head_count_kv").unwrap_or(num_attention_heads);
        let intermediate_size = get_u("feed_forward_length").unwrap_or(5632);
        let max_position_embeddings = get_u("context_length").unwrap_or(4096);
        let rope_theta = get_f("rope.freq_base").unwrap_or(10000.0);
        let rms_norm_eps = get_f("attention.layer_norm_rms_epsilon").unwrap_or(1e-5);
        let num_routed_experts = get_u("expert_count").unwrap_or(64);
        let num_experts_per_tok = get_u("expert_used_count").unwrap_or(6);

        let mut config = serde_json::json!({
            "model_type": model_type,
            "hidden_size": hidden_size,
            "num_hidden_layers": num_hidden_layers,
            "num_attention_heads": num_attention_heads,
            "num_key_value_heads": num_key_value_heads,
            "intermediate_size": intermediate_size,
            "max_position_embeddings": max_position_embeddings,
            "rope_theta": rope_theta,
            "rms_norm_eps": rms_norm_eps,
            "vocab_size": 102400,
        });

        if let Some(obj) = config.as_object_mut() {
            if model_type == "mixtral" {
                obj.insert(
                    "num_local_experts".into(),
                    serde_json::json!(num_routed_experts),
                );
                obj.insert(
                    "num_experts_per_tok".into(),
                    serde_json::json!(num_experts_per_tok),
                );
            } else if model_type.starts_with("deepseek") {
                obj.insert(
                    "n_routed_experts".into(),
                    serde_json::json!(num_routed_experts),
                );
                obj.insert(
                    "num_experts_per_tok".into(),
                    serde_json::json!(num_experts_per_tok),
                );
                obj.insert(
                    "moe_intermediate_size".into(),
                    serde_json::json!(intermediate_size),
                );
                obj.insert("q_lora_rank".into(), serde_json::json!(1536));
                obj.insert("kv_lora_rank".into(), serde_json::json!(512));
                obj.insert("qk_nope_head_dim".into(), serde_json::json!(128));
                obj.insert("qk_rope_head_dim".into(), serde_json::json!(64));
                obj.insert("v_head_dim".into(), serde_json::json!(128));
            } else if model_type.starts_with("qwen") {
                obj.insert("num_experts".into(), serde_json::json!(num_routed_experts));
                obj.insert(
                    "num_experts_per_tok".into(),
                    serde_json::json!(num_experts_per_tok),
                );
                obj.insert(
                    "moe_intermediate_size".into(),
                    serde_json::json!(intermediate_size),
                );
            }
        }

        Ok(config)
    }
}

fn read_string<R: Read>(r: &mut R) -> Result<String> {
    let mut buf8 = [0u8; 8];
    r.read_exact(&mut buf8)?;
    let len = u64::from_le_bytes(buf8) as usize;
    let mut str_bytes = vec![0u8; len];
    r.read_exact(&mut str_bytes)?;
    String::from_utf8(str_bytes).map_err(|e| EngineError::Other(format!("bad utf-8 string: {e}")))
}

fn read_metadata_value<R: Read>(r: &mut R) -> Result<GgufMetadataValue> {
    let mut buf4 = [0u8; 4];
    r.read_exact(&mut buf4)?;
    let type_id = u32::from_le_bytes(buf4);
    read_metadata_value_of_type(r, type_id)
}

fn read_metadata_value_of_type<R: Read>(r: &mut R, type_id: u32) -> Result<GgufMetadataValue> {
    let mut buf1 = [0u8; 1];
    let mut buf2 = [0u8; 2];
    let mut buf4 = [0u8; 4];
    let mut buf8 = [0u8; 8];

    match type_id {
        0 => {
            r.read_exact(&mut buf1)?;
            Ok(GgufMetadataValue::Uint8(buf1[0]))
        }
        1 => {
            r.read_exact(&mut buf1)?;
            Ok(GgufMetadataValue::Int8(buf1[0] as i8))
        }
        2 => {
            r.read_exact(&mut buf2)?;
            Ok(GgufMetadataValue::Uint16(u16::from_le_bytes(buf2)))
        }
        3 => {
            r.read_exact(&mut buf2)?;
            Ok(GgufMetadataValue::Int16(i16::from_le_bytes(buf2)))
        }
        4 => {
            r.read_exact(&mut buf4)?;
            Ok(GgufMetadataValue::Uint32(u32::from_le_bytes(buf4)))
        }
        5 => {
            r.read_exact(&mut buf4)?;
            Ok(GgufMetadataValue::Int32(i32::from_le_bytes(buf4)))
        }
        6 => {
            r.read_exact(&mut buf4)?;
            Ok(GgufMetadataValue::Float32(f32::from_le_bytes(buf4)))
        }
        7 => {
            r.read_exact(&mut buf1)?;
            Ok(GgufMetadataValue::Bool(buf1[0] != 0))
        }
        8 => {
            let s = read_string(r)?;
            Ok(GgufMetadataValue::String(s))
        }
        9 => {
            r.read_exact(&mut buf4)?;
            let elem_type = u32::from_le_bytes(buf4);
            r.read_exact(&mut buf8)?;
            let count = u64::from_le_bytes(buf8) as usize;
            let mut list = Vec::with_capacity(count);
            for _ in 0..count {
                list.push(read_metadata_value_of_type(r, elem_type)?);
            }
            Ok(GgufMetadataValue::Array(list))
        }
        10 => {
            r.read_exact(&mut buf8)?;
            Ok(GgufMetadataValue::Uint64(u64::from_le_bytes(buf8)))
        }
        11 => {
            r.read_exact(&mut buf8)?;
            Ok(GgufMetadataValue::Int64(i64::from_le_bytes(buf8)))
        }
        12 => {
            r.read_exact(&mut buf8)?;
            Ok(GgufMetadataValue::Float64(f64::from_le_bytes(buf8)))
        }
        other => Err(EngineError::Other(format!(
            "unknown GGUF metadata type {other}"
        ))),
    }
}

fn map_expert_tensor_name(raw: &str, expert: usize, arch: &str) -> String {
    if raw.starts_with("model.layers.") {
        return raw.to_string();
    }
    // GGUF format: blk.{layer}.ffn_gate_exps.weight, blk.{layer}.ffn_up_exps.weight, blk.{layer}.ffn_down_exps.weight
    let parts: Vec<&str> = raw.split('.').collect();
    let layer = if parts.len() >= 2 && parts[0] == "blk" {
        parts[1]
    } else {
        "0"
    };

    if arch == "mixtral" {
        if raw.contains("ffn_gate_exps") {
            format!("model.layers.{layer}.block_sparse_moe.experts.{expert}.w1.weight")
        } else if raw.contains("ffn_up_exps") {
            format!("model.layers.{layer}.block_sparse_moe.experts.{expert}.w3.weight")
        } else if raw.contains("ffn_down_exps") {
            format!("model.layers.{layer}.block_sparse_moe.experts.{expert}.w2.weight")
        } else {
            format!("model.layers.{layer}.block_sparse_moe.experts.{expert}.{raw}")
        }
    } else {
        // DeepSeek / Qwen
        if raw.contains("ffn_gate_exps") {
            format!("model.layers.{layer}.mlp.experts.{expert}.gate_proj.weight")
        } else if raw.contains("ffn_up_exps") {
            format!("model.layers.{layer}.mlp.experts.{expert}.up_proj.weight")
        } else if raw.contains("ffn_down_exps") {
            format!("model.layers.{layer}.mlp.experts.{expert}.down_proj.weight")
        } else {
            format!("model.layers.{layer}.mlp.experts.{expert}.{raw}")
        }
    }
}

fn map_standard_tensor_name(raw: &str, arch: &str) -> String {
    if raw.starts_with("model.") || raw == "lm_head.weight" {
        return raw.to_string();
    }

    if raw == "token_embd.weight" {
        return "model.embed_tokens.weight".to_string();
    }
    if raw == "output_norm.weight" {
        return "model.norm.weight".to_string();
    }
    if raw == "output.weight" {
        return "lm_head.weight".to_string();
    }

    let parts: Vec<&str> = raw.split('.').collect();
    if parts.len() >= 3 && parts[0] == "blk" {
        let layer = parts[1];
        let sub = parts[2];
        match sub {
            "attn_q" => format!("model.layers.{layer}.self_attn.q_proj.weight"),
            "attn_k" => format!("model.layers.{layer}.self_attn.k_proj.weight"),
            "attn_v" => format!("model.layers.{layer}.self_attn.v_proj.weight"),
            "attn_output" => format!("model.layers.{layer}.self_attn.o_proj.weight"),
            "attn_norm" => format!("model.layers.{layer}.input_layernorm.weight"),
            "ffn_norm" => format!("model.layers.{layer}.post_attention_layernorm.weight"),
            "ffn_gate_inp" => {
                if arch == "mixtral" {
                    format!("model.layers.{layer}.block_sparse_moe.gate.weight")
                } else {
                    format!("model.layers.{layer}.mlp.gate.weight")
                }
            }
            "ffn_gate_shexp" => {
                format!("model.layers.{layer}.mlp.shared_experts.gate_proj.weight")
            }
            "ffn_up_shexp" => format!("model.layers.{layer}.mlp.shared_experts.up_proj.weight"),
            "ffn_down_shexp" => {
                format!("model.layers.{layer}.mlp.shared_experts.down_proj.weight")
            }
            "ffn_gate" => format!("model.layers.{layer}.mlp.gate_proj.weight"),
            "ffn_up" => format!("model.layers.{layer}.mlp.up_proj.weight"),
            "ffn_down" => format!("model.layers.{layer}.mlp.down_proj.weight"),
            _ => format!("model.layers.{layer}.{sub}"),
        }
    } else {
        raw.to_string()
    }
}

impl TensorSource for GgufReader {
    fn tensor_names(&self) -> Vec<String> {
        self.tensors.keys().cloned().collect()
    }

    fn info(&self, name: &str) -> Result<TensorInfo> {
        let entry = self
            .tensors
            .get(name)
            .ok_or_else(|| EngineError::TensorNotFound(name.to_string()))?;
        let numel = entry.shape.iter().product::<usize>();
        let nbytes = (numel * 4) as u64; // Logical f32 bytes for planning
        Ok(TensorInfo {
            dtype: Dtype::F32,
            shape: entry.shape.clone(),
            offset: entry.file_offset,
            nbytes,
        })
    }

    fn read_f32(&self, name: &str) -> Result<Tensor> {
        let entry = self
            .tensors
            .get(name)
            .ok_or_else(|| EngineError::TensorNotFound(name.to_string()))?;
        let total_elements = entry.shape.iter().product::<usize>();
        let block_size = entry.dtype.block_size();
        let type_size = entry.dtype.type_size();
        let total_bytes = (total_elements / block_size) * type_size;

        let mut raw = vec![0u8; total_bytes];
        #[cfg(unix)]
        self.file.read_exact_at(&mut raw, entry.file_offset)?;
        #[cfg(not(unix))]
        {
            let mut f = File::open(&self.path)?;
            f.seek(SeekFrom::Start(entry.file_offset))?;
            f.read_exact(&mut raw)?;
        }

        let mut data = vec![0f32; total_elements];
        dequantize_blocks(entry.dtype, &raw, &mut data)?;
        Ok(Tensor::new(entry.shape.clone(), data))
    }

    fn read_f32_rows(&self, name: &str, row_start: usize, nrows: usize) -> Result<Tensor> {
        let entry = self
            .tensors
            .get(name)
            .ok_or_else(|| EngineError::TensorNotFound(name.to_string()))?;
        if entry.shape.len() != 2 {
            return Err(EngineError::Other(format!(
                "{name}: read_f32_rows requires 2-D tensor, got {:?}",
                entry.shape
            )));
        }
        if row_start + nrows > entry.out_dim {
            return Err(EngineError::Other(format!(
                "{name}: rows {row_start}..{} out of bounds for {}",
                row_start + nrows,
                entry.out_dim
            )));
        }

        let read_offset = entry.file_offset + (row_start * entry.bytes_per_row) as u64;
        let read_len = nrows * entry.bytes_per_row;
        let mut raw = vec![0u8; read_len];

        #[cfg(unix)]
        self.file.read_exact_at(&mut raw, read_offset)?;
        #[cfg(not(unix))]
        {
            let mut f = File::open(&self.path)?;
            f.seek(SeekFrom::Start(read_offset))?;
            f.read_exact(&mut raw)?;
        }

        let mut data = vec![0f32; nrows * entry.in_dim];
        dequantize_blocks(entry.dtype, &raw, &mut data)?;
        Ok(Tensor::new(vec![nrows, entry.in_dim], data))
    }
}

/// Convert a GGUF file directly into a streaming Undertow checkpoint layout.
pub fn convert_gguf<F>(
    src: impl AsRef<Path>,
    dst: impl AsRef<Path>,
    classify_resolver: F,
    opts: &ConvertOptions,
) -> Result<ConvertReport>
where
    F: FnOnce(&Path) -> Result<Box<dyn Fn(&str) -> Disposition + Sync>>,
{
    let src = src.as_ref();
    let dst = dst.as_ref();
    if opts.row_chunk == 0 {
        return Err(EngineError::InvalidConfig("row_chunk must be > 0".into()));
    }
    std::fs::create_dir_all(dst)?;
    let reader = GgufReader::open(src)?;

    // 1. Synthesize config.json if not present
    let config_path = dst.join("config.json");
    if !config_path.exists() {
        let config_json = reader.synthesize_config()?;
        std::fs::write(
            &config_path,
            serde_json::to_vec_pretty(&config_json).expect("valid json"),
        )?;
    }

    // 2. Resolve architecture classifier
    let classifier_box = classify_resolver(dst)?;
    let classify: Classifier = &*classifier_box;

    // 3. Convert from GGUF source
    convert_from_source(&reader, &src.to_string_lossy(), dst, classify, opts)
}

/// Check if a path points to a GGUF file (by extension or 4-byte magic).
pub fn is_gguf_file(path: impl AsRef<Path>) -> bool {
    let p = path.as_ref();
    if let Some(ext) = p.extension().and_then(|s| s.to_str()) {
        if ext.eq_ignore_ascii_case("gguf") {
            return true;
        }
    }
    if let Ok(mut f) = File::open(p) {
        let mut magic = [0u8; 4];
        if f.read_exact(&mut magic).is_ok() {
            return u32::from_le_bytes(magic) == GGUF_MAGIC;
        }
    }
    false
}
