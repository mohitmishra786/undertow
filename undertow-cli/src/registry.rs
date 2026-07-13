//! Architecture dispatch: config.json `model_type` → adapter family.
//!
//! This is the composition root; adapter crates never know about each
//! other, and the core never branches on a model name. Adding a family
//! means adding one arm here.

use std::path::Path;

use anyhow::{bail, Context, Result};
use undertow_convert::Disposition;
use undertow_core::model::LoadOptions;
use undertow_core::Model;

pub fn model_type(dir: &Path) -> Result<String> {
    let path = dir.join("config.json");
    let root: serde_json::Value = serde_json::from_slice(
        &std::fs::read(&path).with_context(|| format!("reading {}", path.display()))?,
    )
    .with_context(|| format!("parsing {}", path.display()))?;
    root.get("model_type")
        .and_then(|v| v.as_str())
        .map(str::to_string)
        .with_context(|| format!("{}: missing model_type", path.display()))
}

/// Families served by the DeepSeek-style adapter (sigmoid noaux_tc + MLA).
const DEEPSEEK_TYPES: &[&str] = &["deepseek_v3", "glm_moe", "glm_moe_dsa", "kimi_k2"];

pub fn load_any(dir: &Path, opts: &LoadOptions) -> Result<Box<dyn Model>> {
    let ty = model_type(dir)?;
    let model: Box<dyn Model> = match ty.as_str() {
        t if DEEPSEEK_TYPES.contains(&t) => {
            Box::new(undertow_deepseek_moe::loader::load_model_with(dir, opts)?)
        }
        "mixtral" => Box::new(undertow_mixtral_moe::load_model_with(dir, opts)?),
        "qwen3_moe" => Box::new(undertow_qwen_moe::load_model_with(dir, opts)?),
        other => bail!(
            "unsupported model_type {other:?} (supported: {DEEPSEEK_TYPES:?}, \"mixtral\", \"qwen3_moe\")"
        ),
    };
    Ok(model)
}

pub fn classifier_for(dir: &Path) -> Result<fn(&str) -> Disposition> {
    let ty = model_type(dir)?;
    Ok(match ty.as_str() {
        t if DEEPSEEK_TYPES.contains(&t) => undertow_deepseek_moe::classify_tensor,
        "mixtral" => undertow_mixtral_moe::classify_tensor,
        "qwen3_moe" => undertow_qwen_moe::classify_tensor,
        other => bail!("unsupported model_type {other:?} for conversion"),
    })
}

/// Concrete DeepSeek-family model (MTP speculation needs family API).
pub fn as_deepseek(
    dir: &Path,
    opts: &LoadOptions,
) -> Result<undertow_deepseek_moe::DeepseekMoeModel> {
    let ty = model_type(dir)?;
    if !DEEPSEEK_TYPES.contains(&ty.as_str()) {
        bail!("--mtp is only available for DeepSeek-family models, got {ty:?}");
    }
    Ok(undertow_deepseek_moe::loader::load_model_with(dir, opts)?)
}
