//! Correctness of every adapter family's forward pass against its
//! `transformers` golden reference.
//!
//! Each fixture is a tiny random-weight checkpoint with the real
//! architecture, generated deterministically by `engine_bench` and run
//! once through the upstream implementation (`tools/make_reference.py`).
//! Passing here means attention (MLA and GQA variants), RoPE (interleaved
//! and neox), routers (noaux_tc sigmoid and softmax top-k), shared
//! experts, dense layers and q/k norms all agree with upstream,
//! token-exactly.

use std::path::PathBuf;

use engine_core::Model;
use serde::Deserialize;

#[derive(Deserialize)]
struct Reference {
    prompt_ids: Vec<usize>,
    full_ids: Vec<usize>,
    tf_pred: Vec<usize>,
    tf_logits: Vec<Vec<f32>>,
}

fn fixture_dir(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("fixtures/{name}"))
}

fn load_reference(name: &str) -> Reference {
    let bytes = std::fs::read(fixture_dir(name).join("reference.json"))
        .expect("reference.json missing — see engine-bench/tools/make_reference.py");
    serde_json::from_slice(&bytes).expect("bad reference.json")
}

fn load_family(name: &str) -> Box<dyn Model> {
    let dir = fixture_dir(name);
    match name {
        "oracle-tiny" => Box::new(deepseek_moe::loader::load_model(dir).unwrap()),
        "oracle-mixtral-tiny" => Box::new(mixtral_moe::load_model(dir).unwrap()),
        "oracle-qwen-tiny" => Box::new(qwen_moe::load_model(dir).unwrap()),
        other => panic!("unknown fixture {other}"),
    }
}

const FAMILIES: &[&str] = &["oracle-tiny", "oracle-mixtral-tiny", "oracle-qwen-tiny"];

#[test]
fn teacher_forcing_matches_transformers() {
    for name in FAMILIES {
        let reference = load_reference(name);
        let model = load_family(name);
        let logits = model.forward(&reference.full_ids).expect("forward");

        let seq = reference.full_ids.len();
        assert_eq!(logits.shape, vec![seq, model.vocab_size()], "{name}");

        let mut max_err = 0.0f32;
        for (s, ref_row) in reference.tf_logits.iter().enumerate() {
            for (ours, theirs) in logits.row(s).iter().zip(ref_row) {
                max_err = max_err.max((ours - theirs).abs());
            }
        }
        eprintln!("{name}: max logit error vs transformers: {max_err:e}");
        assert!(max_err < 1e-3, "{name}: max logit error {max_err}");

        // Token-exact: argmax at every position (the tf_pred check).
        let ours_pred: Vec<usize> = (0..seq)
            .map(|s| engine_core::sample::argmax(logits.row(s)))
            .collect();
        assert_eq!(ours_pred, reference.tf_pred, "{name}: tf argmax mismatch");
    }
}

#[test]
fn greedy_decode_matches_transformers() {
    for name in FAMILIES {
        let reference = load_reference(name);
        let model = load_family(name);
        let max_new = reference.full_ids.len() - reference.prompt_ids.len();
        let ours = engine_core::greedy_decode(&*model, &reference.prompt_ids, max_new, &[])
            .expect("decode");
        assert_eq!(ours, reference.full_ids, "{name}: greedy mismatch");
    }
}

/// Golden references are only valid for the exact fixture weights. The
/// generators are platform-deterministic, so regeneration must reproduce
/// the checked-in fixtures byte-for-byte; drift means someone changed a
/// generator without regenerating references.
#[test]
fn fixtures_match_generators() {
    let tmp = tempfile::tempdir().unwrap();
    type GenFn<'a> = &'a dyn Fn(&std::path::Path);
    let cases: &[(&str, GenFn)] = &[
        ("oracle-tiny", &|d| {
            engine_bench::generate_oracle(d, &engine_bench::OracleSpec::default()).unwrap()
        }),
        ("oracle-mixtral-tiny", &|d| {
            engine_bench::generate_mixtral_oracle(d, 20260712).unwrap()
        }),
        ("oracle-qwen-tiny", &|d| {
            engine_bench::generate_qwen_oracle(d, 20260712).unwrap()
        }),
    ];
    for (name, gen) in cases {
        let out = tmp.path().join(name);
        gen(&out);
        let regenerated = std::fs::read(out.join("model.safetensors")).unwrap();
        let fixture = std::fs::read(fixture_dir(name).join("model.safetensors")).unwrap();
        assert!(
            regenerated == fixture,
            "{name}: fixture differs from generator output — regenerate the \
             fixture AND reference.json (see tools/make_reference.py)"
        );
    }
}
