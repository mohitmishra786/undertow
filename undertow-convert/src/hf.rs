use std::collections::HashMap;
use std::sync::Arc;

use reqwest::blocking::Client;
use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION, RANGE};
use undertow_core::{EngineError, Result, Tensor};
use undertow_io::{decode_f32, parse_header, TensorInfo};

use crate::TensorSource;

const MAX_HEADER_LEN: u64 = 100 * 1024 * 1024;

/// Configuration for connecting to Hugging Face Hub.
#[derive(Debug, Clone)]
pub struct HfConfig {
    pub repo_id: String,
    pub revision: String,
    pub token: Option<String>,
    pub endpoint: String,
}

impl HfConfig {
    pub fn new(repo_id: impl Into<String>) -> Self {
        let endpoint = std::env::var("HF_ENDPOINT")
            .unwrap_or_else(|_| "https://huggingface.co".to_string())
            .trim_end_matches('/')
            .to_string();
        let token = std::env::var("HF_TOKEN").ok();
        Self {
            repo_id: repo_id.into(),
            revision: "main".to_string(),
            token,
            endpoint,
        }
    }

    pub fn with_token(mut self, token: Option<String>) -> Self {
        if token.is_some() {
            self.token = token;
        }
        self
    }

    pub fn with_revision(mut self, revision: impl Into<String>) -> Self {
        self.revision = revision.into();
        self
    }

    pub fn with_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.endpoint = endpoint.into().trim_end_matches('/').to_string();
        self
    }
}

pub struct HfClient {
    pub config: HfConfig,
    client: Client,
}

impl HfClient {
    pub fn new(config: HfConfig) -> Result<Self> {
        let client = Client::builder()
            .build()
            .map_err(|e| EngineError::Other(format!("failed to initialize HTTP client: {e}")))?;
        Ok(Self { config, client })
    }

    pub fn file_url(&self, filename: &str) -> String {
        format!(
            "{}/{}/resolve/{}/{}",
            self.config.endpoint, self.config.repo_id, self.config.revision, filename
        )
    }

    fn headers(&self) -> HeaderMap {
        let mut headers = HeaderMap::new();
        if let Some(ref tok) = self.config.token {
            if let Ok(val) = HeaderValue::from_str(&format!("Bearer {tok}")) {
                headers.insert(AUTHORIZATION, val);
            }
        }
        headers
    }

    /// Fetch a full file if it exists. Returns `None` if 404.
    pub fn fetch_file(&self, filename: &str) -> Result<Option<Vec<u8>>> {
        let url = self.file_url(filename);
        let resp = self
            .client
            .get(&url)
            .headers(self.headers())
            .send()
            .map_err(|e| EngineError::Other(format!("HTTP error connecting to {url}: {e}")))?;

        if resp.status() == reqwest::StatusCode::NOT_FOUND {
            return Ok(None);
        }
        if !resp.status().is_success() {
            return Err(EngineError::Other(format!(
                "HTTP {} fetching {url}",
                resp.status()
            )));
        }
        let bytes = resp
            .bytes()
            .map_err(|e| EngineError::Other(format!("error reading body from {url}: {e}")))?;
        Ok(Some(bytes.to_vec()))
    }

    /// Fetch a byte range `[start, end]` (inclusive).
    pub fn fetch_range(&self, filename: &str, start: u64, end: u64) -> Result<Vec<u8>> {
        let url = self.file_url(filename);
        let range_val = format!("bytes={start}-{end}");
        let mut headers = self.headers();
        headers.insert(RANGE, HeaderValue::from_str(&range_val).unwrap());

        let resp = self.client.get(&url).headers(headers).send().map_err(|e| {
            EngineError::Other(format!(
                "HTTP error fetching range {range_val} from {url}: {e}"
            ))
        })?;

        let status = resp.status();
        if status != reqwest::StatusCode::PARTIAL_CONTENT && status != reqwest::StatusCode::OK {
            return Err(EngineError::Other(format!(
                "HTTP {status} fetching byte range {range_val} from {url}"
            )));
        }

        let bytes = resp
            .bytes()
            .map_err(|e| EngineError::Other(format!("error reading body from {url}: {e}")))?;

        let expected_len = (end - start + 1) as usize;
        if status == reqwest::StatusCode::PARTIAL_CONTENT {
            if bytes.len() != expected_len {
                return Err(EngineError::Other(format!(
                    "{url}: short range read: expected {expected_len} bytes, got {}",
                    bytes.len()
                )));
            }
            Ok(bytes.to_vec())
        } else {
            // Server ignored Range header and sent full body
            if bytes.len() < (end + 1) as usize {
                return Err(EngineError::Other(format!(
                    "{url}: response too short for range {start}-{end}: len {}",
                    bytes.len()
                )));
            }
            Ok(bytes[start as usize..=end as usize].to_vec())
        }
    }

    /// Fetch file size via HEAD or range probe.
    pub fn fetch_file_len(&self, filename: &str) -> Result<u64> {
        let url = self.file_url(filename);
        let resp = self
            .client
            .head(&url)
            .headers(self.headers())
            .send()
            .map_err(|e| EngineError::Other(format!("HTTP HEAD {url}: {e}")))?;

        if resp.status().is_success() {
            if let Some(val) = resp.headers().get(reqwest::header::CONTENT_LENGTH) {
                if let Ok(s) = val.to_str() {
                    if let Ok(len) = s.trim().parse::<u64>() {
                        return Ok(len);
                    }
                }
            }
        }

        // Fallback: range request probe
        let mut headers = self.headers();
        headers.insert(RANGE, HeaderValue::from_static("bytes=0-0"));
        let resp = self
            .client
            .get(&url)
            .headers(headers)
            .send()
            .map_err(|e| EngineError::Other(format!("HTTP range probe {url}: {e}")))?;

        if let Some(cr) = resp.headers().get(reqwest::header::CONTENT_RANGE) {
            if let Ok(cr_str) = cr.to_str() {
                // Format: "bytes 0-0/1234567"
                if let Some(total) = cr_str.split('/').nth(1) {
                    if let Ok(len) = total.trim().parse::<u64>() {
                        return Ok(len);
                    }
                }
            }
        }

        Err(EngineError::Other(format!(
            "could not determine file length for {url}"
        )))
    }

    /// Read safetensors header from remote shard.
    pub fn fetch_shard_header(&self, filename: &str) -> Result<HashMap<String, TensorInfo>> {
        let file_len = self.fetch_file_len(filename)?;
        if file_len < 8 {
            return Err(EngineError::Other(format!(
                "{filename}: file too small for safetensors header"
            )));
        }

        // Fetch up to 1MB or file_len to grab the header in one request
        let probe_len = (1024 * 1024).min(file_len);
        let probe = self.fetch_range(filename, 0, probe_len - 1)?;
        if probe.len() < 8 {
            return Err(EngineError::Other(format!(
                "{filename}: failed to read header prefix"
            )));
        }

        let header_len = u64::from_le_bytes(probe[0..8].try_into().unwrap());
        if header_len > MAX_HEADER_LEN {
            return Err(EngineError::Other(format!(
                "{filename}: header length {header_len} exceeds max {MAX_HEADER_LEN}"
            )));
        }

        let header_bytes = if 8 + header_len <= probe.len() as u64 {
            probe[8..8 + header_len as usize].to_vec()
        } else {
            self.fetch_range(filename, 8, 8 + header_len - 1)?
        };

        parse_header(&header_bytes, 8 + header_len, file_len, filename)
    }
}

pub struct HfRemoteSource {
    client: Arc<HfClient>,
    tensor_index: HashMap<String, (String, TensorInfo)>,
}

impl HfRemoteSource {
    pub fn open(config: HfConfig) -> Result<Self> {
        let client = HfClient::new(config)?;
        Self::with_client(client)
    }

    pub fn with_client(client: HfClient) -> Result<Self> {
        let client = Arc::new(client);
        let mut tensor_index = HashMap::new();

        // Check model.safetensors.index.json
        if let Some(index_bytes) = client.fetch_file("model.safetensors.index.json")? {
            let root: serde_json::Value = serde_json::from_slice(&index_bytes)
                .map_err(|e| EngineError::Other(format!("bad remote index json: {e}")))?;
            let weight_map = root
                .get("weight_map")
                .and_then(|m| m.as_object())
                .ok_or_else(|| EngineError::Other("missing weight_map in remote index".into()))?;

            let mut shards = HashMap::new();
            for (tensor, shard_val) in weight_map {
                let shard_file = shard_val
                    .as_str()
                    .ok_or_else(|| EngineError::Other("bad weight_map value".into()))?;
                let shard_header = if let Some(h) = shards.get(shard_file) {
                    h
                } else {
                    let h = client.fetch_shard_header(shard_file)?;
                    shards.insert(shard_file.to_string(), h);
                    shards.get(shard_file).unwrap()
                };
                let info = shard_header
                    .get(tensor)
                    .ok_or_else(|| EngineError::TensorNotFound(tensor.clone()))?;
                tensor_index.insert(tensor.clone(), (shard_file.to_string(), info.clone()));
            }
        } else {
            // Check single model.safetensors
            let shard_file = "model.safetensors";
            let shard_header = client.fetch_shard_header(shard_file)?;
            for (tensor, info) in shard_header {
                tensor_index.insert(tensor, (shard_file.to_string(), info));
            }
        }

        if tensor_index.is_empty() {
            return Err(EngineError::Other(
                "no safetensors found in remote repository".into(),
            ));
        }

        Ok(Self {
            client,
            tensor_index,
        })
    }
}

impl TensorSource for HfRemoteSource {
    fn tensor_names(&self) -> Vec<String> {
        self.tensor_index.keys().cloned().collect()
    }

    fn info(&self, name: &str) -> Result<TensorInfo> {
        self.tensor_index
            .get(name)
            .map(|(_, info)| info.clone())
            .ok_or_else(|| EngineError::TensorNotFound(name.to_string()))
    }

    fn read_f32(&self, name: &str) -> Result<Tensor> {
        let (shard, info) = self
            .tensor_index
            .get(name)
            .ok_or_else(|| EngineError::TensorNotFound(name.to_string()))?;
        let raw = self
            .client
            .fetch_range(shard, info.offset, info.offset + info.nbytes - 1)?;
        let ctx = format!("{}/{}", self.client.config.repo_id, shard);
        let data = decode_f32(&raw, info.dtype, name, &ctx)?;
        Ok(Tensor::new(info.shape.clone(), data))
    }

    fn read_f32_rows(&self, name: &str, row_start: usize, nrows: usize) -> Result<Tensor> {
        let (shard, info) = self
            .tensor_index
            .get(name)
            .ok_or_else(|| EngineError::TensorNotFound(name.to_string()))?;
        if info.shape.len() != 2 {
            return Err(EngineError::Other(format!(
                "{name}: read_f32_rows requires 2-D tensor, got {:?}",
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
        let nbytes = (nrows * in_dim * elem) as u64;
        let raw = self
            .client
            .fetch_range(shard, offset, offset + nbytes - 1)?;
        let ctx = format!("{}/{}", self.client.config.repo_id, shard);
        let data = decode_f32(&raw, info.dtype, name, &ctx)?;
        Ok(Tensor::new(vec![nrows, in_dim], data))
    }
}
