//! Fast-int8 quality gate, in its own test binary because the switch is
//! process-global: no other tests share this process, so flipping it here
//! cannot perturb the exact-numerics assertions elsewhere.

use std::path::PathBuf;

use undertow_core::sample::argmax;
use undertow_core::QuantFormat;
use undertow_deepseek_moe::loader::{load_model_with, LoadOptions, StoreChoice};

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/oracle-tiny")
}

#[derive(serde::Deserialize)]
struct Reference {
    full_ids: Vec<usize>,
    tf_pred: Vec<usize>,
}

/// With integer activation quantization on an int8 model, teacher-forcing
/// argmax agreement must stay near the exact-activation floor. Random
/// weights are the worst case; trained models fare better.
#[test]
fn fast_int8_keeps_oracle_agreement() {
    let reference: Reference =
        serde_json::from_slice(&std::fs::read(fixture_dir().join("reference.json")).unwrap())
            .unwrap();
    let tmp = tempfile::tempdir().unwrap();
    let opts = undertow_convert::ConvertOptions {
        expert_format: QuantFormat::Int8,
        dense_format: QuantFormat::Int8,
        ..Default::default()
    };
    undertow_convert::convert(
        fixture_dir(),
        tmp.path(),
        &undertow_deepseek_moe::classify_tensor,
        &opts,
    )
    .unwrap();
    let model = load_model_with(
        tmp.path(),
        &LoadOptions {
            store: StoreChoice::Resident,
            ..Default::default()
        },
    )
    .unwrap();

    undertow_core::set_fast_int8(true);
    let logits = model.forward(&reference.full_ids).unwrap();
    undertow_core::set_fast_int8(false);

    let agree = (0..reference.full_ids.len())
        .filter(|&s| argmax(logits.row(s)) == reference.tf_pred[s])
        .count();
    let rate = agree as f64 / reference.full_ids.len() as f64;
    eprintln!("fast-int8 tf agreement: {rate:.2}");
    assert!(rate >= 0.75, "fast-int8 agreement {rate:.2} below floor");
}
