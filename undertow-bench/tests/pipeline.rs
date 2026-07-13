//! End-to-end pipeline tests over the oracle fixture: conversion,
//! quantized inference, disk streaming under memory pressure, and
//! incremental-decode equivalence.
//!
//! Everything here is deterministic (fixed fixture, integer quantization,
//! ordered f32 arithmetic), so assertions are exact where the design says
//! they must be exact, and threshold-based only where quantization
//! genuinely loses information.

use std::path::{Path, PathBuf};

use undertow_core::sample::argmax;
use undertow_core::QuantFormat;
use undertow_deepseek_moe::loader::{load_model_with, LoadOptions, StoreChoice};

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/oracle-tiny")
}

#[derive(serde::Deserialize)]
struct Reference {
    prompt_ids: Vec<usize>,
    full_ids: Vec<usize>,
    tf_pred: Vec<usize>,
}

fn load_reference() -> Reference {
    serde_json::from_slice(&std::fs::read(fixture_dir().join("reference.json")).unwrap()).unwrap()
}

fn convert_fixture(dst: &Path, experts: QuantFormat, dense: QuantFormat) {
    let opts = undertow_convert::ConvertOptions {
        expert_format: experts,
        dense_format: dense,
        row_chunk: 7, // deliberately awkward chunk size
        force: false,
    };
    undertow_convert::convert(
        fixture_dir(),
        dst,
        &undertow_deepseek_moe::classify_tensor,
        &opts,
    )
    .unwrap();
}

fn resident() -> LoadOptions {
    LoadOptions {
        store: StoreChoice::Resident,
        ..Default::default()
    }
}

fn streaming(budget: u64) -> LoadOptions {
    LoadOptions {
        store: StoreChoice::DiskStreaming {
            cache_budget_bytes: Some(budget),
            prefetch_workers: 2,
        },
        ..Default::default()
    }
}

/// Streaming with a starvation-level cache budget must produce logits
/// bit-identical to the resident store: eviction may cost time, never
/// correctness.
#[test]
fn streaming_under_pressure_is_bit_identical_to_resident() {
    let reference = load_reference();
    let tmp = tempfile::tempdir().unwrap();
    convert_fixture(tmp.path(), QuantFormat::Int4, QuantFormat::Int8);

    let resident_model = load_model_with(tmp.path(), &resident()).unwrap();
    // ~2 experts worth of cache: constant eviction across 16 experts/layer.
    let starved_model = load_model_with(tmp.path(), &streaming(20_000)).unwrap();

    let a = resident_model.forward(&reference.full_ids).unwrap();
    let b = starved_model.forward(&reference.full_ids).unwrap();
    assert_eq!(a.data, b.data, "storage tier must not affect numerics");

    // Expert batching fetches each unique expert once per forward, so a
    // single pass shows only cold misses. Starvation is visible across
    // passes: a second identical forward must re-read nearly everything
    // because the budget cannot retain the working set.
    let first = starved_model.store.stats();
    assert!(first.misses > 20, "cold pass must miss, stats: {first:?}");
    let b2 = starved_model.forward(&reference.full_ids).unwrap();
    assert_eq!(a.data, b2.data, "second pass must match too");
    let second = starved_model.store.stats();
    let refetched = second.misses - first.misses;
    assert!(
        refetched as f64 >= first.misses as f64 * 0.8,
        "starved cache retained too much: {first:?} then {second:?}"
    );
}

/// Incremental decode (prefill + one token at a time, absorbed attention)
/// must agree with the one-shot forward on next-token argmax at every
/// step, for f32 and for quantized weights.
#[test]
fn incremental_decode_matches_full_forward() {
    let reference = load_reference();
    let tmp = tempfile::tempdir().unwrap();
    convert_fixture(tmp.path(), QuantFormat::Int8, QuantFormat::Int8);

    for dir in [fixture_dir(), tmp.path().to_path_buf()] {
        let model = load_model_with(&dir, &resident()).unwrap();
        let full = model.forward(&reference.full_ids).unwrap();

        let mut session = model.session();
        let split = reference.prompt_ids.len();
        let prefill_logits = session.prefill(&reference.full_ids[..split]).unwrap();
        assert_eq!(
            argmax(prefill_logits.row(split - 1)),
            argmax(full.row(split - 1)),
            "{}: prefill boundary",
            dir.display()
        );
        for (step, &id) in reference.full_ids[split..].iter().enumerate() {
            let logits = session.decode(id).unwrap();
            let pos = split + step;
            assert_eq!(
                argmax(&logits),
                argmax(full.row(pos)),
                "{}: decode step at position {pos}",
                dir.display()
            );
            let max_err = logits
                .iter()
                .zip(full.row(pos))
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            assert!(max_err < 1e-3, "{}: pos {pos} err {max_err}", dir.display());
        }
    }
}

/// int8 quantization must preserve most teacher-forcing argmax decisions
/// on the oracle, and the greedy continuation must share a prefix with
/// the f32 reference. Random weights are the worst case for quantization
/// (zero redundancy), so these floors are deliberately conservative;
/// regressions in kernels or conversion push them to zero and fail loudly.
#[test]
fn quantized_models_stay_close_to_reference() {
    let reference = load_reference();
    let cases = [(QuantFormat::Int8, 0.85, 6), (QuantFormat::Int4, 0.55, 3)];
    for (fmt, min_agreement, min_prefix) in cases {
        let tmp = tempfile::tempdir().unwrap();
        convert_fixture(tmp.path(), fmt, QuantFormat::Int8);
        let model = load_model_with(tmp.path(), &resident()).unwrap();

        let logits = model.forward(&reference.full_ids).unwrap();
        let agree = (0..reference.full_ids.len())
            .filter(|&s| argmax(logits.row(s)) == reference.tf_pred[s])
            .count();
        let rate = agree as f64 / reference.full_ids.len() as f64;
        assert!(
            rate >= min_agreement,
            "{fmt:?}: tf argmax agreement {rate:.2} below floor {min_agreement}"
        );

        let max_new = reference.full_ids.len() - reference.prompt_ids.len();
        let ours = model
            .greedy_decode(&reference.prompt_ids, max_new, &[])
            .unwrap();
        let prefix = ours
            .iter()
            .zip(&reference.full_ids)
            .take_while(|(a, b)| a == b)
            .count()
            - reference.prompt_ids.len();
        assert!(
            prefix >= min_prefix,
            "{fmt:?}: greedy prefix {prefix} below floor {min_prefix} (got {ours:?})"
        );
    }
}

/// The context window is a hard limit, reported as a typed error.
#[test]
fn context_overflow_is_reported() {
    let model = load_model_with(fixture_dir(), &resident()).unwrap();
    let max = model.cfg.max_position_embeddings;
    let long: Vec<usize> = (0..max + 1).map(|i| i % model.cfg.vocab_size).collect();
    match model.forward(&long) {
        Err(undertow_core::EngineError::ContextOverflow { requested, max: m }) => {
            assert_eq!(requested, max + 1);
            assert_eq!(m, max);
        }
        other => panic!("expected ContextOverflow, got {other:?}"),
    }
    // Session state survives a rejected prefill.
    let mut session = model.session();
    assert!(session.prefill(&long).is_err());
    assert!(session.prefill(&[1, 2, 3]).is_ok());
}

/// Session truncation (chat prefix reuse) must reproduce the same logits
/// as a fresh session over the same tokens.
#[test]
fn truncate_and_refill_matches_fresh_session() {
    let reference = load_reference();
    let model = load_model_with(fixture_dir(), &resident()).unwrap();
    let ids = &reference.full_ids;

    let mut a = model.session();
    a.prefill(&ids[..20]).unwrap();
    a.truncate(10);
    let logits_a = a.prefill(&ids[10..20]).unwrap();

    let mut b = model.session();
    let logits_b = b.prefill(&ids[..20]).unwrap();

    // Same tokens, same positions: identical results row for row.
    let last_a = logits_a.row(9);
    let last_b = logits_b.row(19);
    let max_err = last_a
        .iter()
        .zip(last_b)
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max);
    assert!(max_err < 1e-4, "truncate/refill diverged: {max_err}");
}

/// Concurrent sessions over one shared streaming model must not interfere.
#[test]
fn concurrent_sessions_are_isolated() {
    let reference = load_reference();
    let tmp = tempfile::tempdir().unwrap();
    convert_fixture(tmp.path(), QuantFormat::Int4, QuantFormat::Int8);
    let model = std::sync::Arc::new(load_model_with(tmp.path(), &streaming(50_000)).unwrap());

    let baseline = model.forward(&reference.full_ids).unwrap();
    let mut handles = Vec::new();
    for _ in 0..4 {
        let model = model.clone();
        let ids = reference.full_ids.clone();
        let expect = baseline.data.clone();
        handles.push(std::thread::spawn(move || {
            for _ in 0..3 {
                let got = model.forward(&ids).unwrap();
                assert_eq!(got.data, expect, "concurrent session diverged");
            }
        }));
    }
    for h in handles {
        h.join().unwrap();
    }
}

/// A profile recorded on one run and pinned on the next must remove all
/// misses for the recorded working set, without changing the numerics.
#[test]
fn profile_pinning_eliminates_misses() {
    use undertow_core::Model;
    let reference = load_reference();
    let tmp = tempfile::tempdir().unwrap();
    convert_fixture(tmp.path(), QuantFormat::Int4, QuantFormat::Int8);

    // Record usage on a first run.
    let recorder = load_model_with(tmp.path(), &streaming(1 << 30)).unwrap();
    let baseline = recorder.forward(&reference.full_ids).unwrap();
    let counts: Vec<(usize, usize, u64)> = recorder
        .expert_usage()
        .into_iter()
        .map(|(k, c)| (k.layer, k.expert, c))
        .collect();
    assert!(!counts.is_empty(), "usage must be recorded");
    let profile = undertow_core::ExpertProfile::new("deepseek_moe", counts);

    // Fresh model with the profile pinned: the recorded working set is
    // resident before the first token, so nothing misses.
    let opts = LoadOptions {
        store: StoreChoice::DiskStreaming {
            cache_budget_bytes: Some(1 << 30),
            prefetch_workers: 0,
        },
        cache_policy: undertow_core::model::CachePolicy::Weighted,
        pin_profile: Some(profile),
        pin_budget_bytes: Some(1 << 30),
        kv_budget_bytes: None,
    };
    let pinned = load_model_with(tmp.path(), &opts).unwrap();
    let logits = pinned.forward(&reference.full_ids).unwrap();
    assert_eq!(
        logits.data, baseline.data,
        "pinning must not change numerics"
    );
    let s = pinned.store.stats();
    assert_eq!(s.misses, 0, "pinned working set must not miss: {s:?}");
    assert!(s.hits > 0);
}

/// MTP speculation is lossless by construction: greedy output must be
/// identical with the draft head on or off, whatever the (random, mostly
/// wrong) draft head predicts. Exercised on f32 and on a converted int8
/// checkpoint, streaming.
#[test]
fn mtp_speculation_is_lossless() {
    let reference = load_reference();
    let tmp = tempfile::tempdir().unwrap();
    convert_fixture(tmp.path(), QuantFormat::Int8, QuantFormat::Int8);

    for dir in [fixture_dir(), tmp.path().to_path_buf()] {
        let model = load_model_with(&dir, &streaming(1 << 30)).unwrap();
        assert!(model.mtp.is_some(), "oracle must ship an MTP head");
        let max_new = 24;
        let plain = model
            .greedy_decode(&reference.prompt_ids, max_new, &[])
            .unwrap();

        let mut speculated = reference.prompt_ids.clone();
        let (produced, stats) = undertow_deepseek_moe::generate_greedy_mtp(
            &model,
            &reference.prompt_ids,
            max_new,
            &[],
            |id| {
                speculated.push(id);
                true
            },
        )
        .unwrap();
        assert_eq!(produced, max_new, "{}", dir.display());
        assert_eq!(
            speculated,
            plain,
            "{}: MTP changed greedy output (stats {stats:?})",
            dir.display()
        );
        eprintln!(
            "{}: mtp acceptance {}/{}",
            dir.display(),
            stats.accepted,
            stats.drafted
        );
    }
}

/// Drive the speculative loop with a perfect draft (the main model's own
/// greedy continuation) so the acceptance branch is fully exercised:
/// every draft accepted, two tokens per verify forward, output unchanged.
#[test]
fn speculation_acceptance_path_is_exact() {
    let reference = load_reference();
    let model = load_model_with(fixture_dir(), &resident()).unwrap();
    let max_new = 20;
    let plain = model
        .greedy_decode(&reference.prompt_ids, max_new, &[])
        .unwrap();

    let emitted = std::cell::RefCell::new(Vec::<usize>::new());
    let prompt = reference.prompt_ids.clone();
    // Shadow session mirroring the main one through the same incremental
    // code path, so its argmax is bit-identical to what verification will
    // compute (a fresh full forward can flip near-ties by ~1e-6 and cause
    // spurious rejections).
    let shadow = std::cell::RefCell::new(model.session());
    shadow.borrow_mut().prefill(&prompt).unwrap();
    let (produced, stats) = undertow_deepseek_moe::speculative_loop(
        &model,
        &reference.prompt_ids,
        max_new,
        &[],
        |id| {
            emitted.borrow_mut().push(id);
            true
        },
        |_, next| {
            // `emitted` already ends with `next` (the loop reports the
            // token before drafting), so the shadow's target sequence is
            // exactly prompt + emitted.
            let mut ids = prompt.clone();
            ids.extend_from_slice(&emitted.borrow());
            assert_eq!(*ids.last().unwrap(), next);
            let mut sh = shadow.borrow_mut();
            let fed = sh.position();
            let logits = sh.prefill(&ids[fed..])?;
            Ok(argmax(logits.row(ids.len() - fed - 1)))
        },
    )
    .unwrap();
    let emitted = emitted.into_inner();
    assert_eq!(produced, max_new);
    assert_eq!(
        stats.accepted, stats.drafted,
        "perfect drafts must all accept"
    );
    assert!(stats.drafted > 0);
    let mut full = reference.prompt_ids.clone();
    full.extend_from_slice(&emitted);
    assert_eq!(full, plain, "accepted speculation changed greedy output");
}

/// Running out of context mid-generation ends the stream cleanly with the
/// tokens produced so far, instead of surfacing an error.
#[test]
fn generation_stops_cleanly_at_context_end() {
    let model = load_model_with(fixture_dir(), &resident()).unwrap();
    let max = undertow_core::Model::max_context(&model);
    // Prompt that nearly fills the window, then ask for far more than fits.
    let prompt: Vec<usize> = (0..max - 4).map(|i| i % model.cfg.vocab_size).collect();
    let out = model.greedy_decode(&prompt, 1000, &[]).unwrap();
    assert!(out.len() > prompt.len(), "must produce at least one token");
    assert!(out.len() <= max + 1, "must stop at the window");
}

/// A session KV budget lowers the usable context and reports it through
/// max_context, and overflow against the lowered bound is typed.
#[test]
fn kv_budget_lowers_effective_context() {
    let opts = LoadOptions {
        store: StoreChoice::Resident,
        kv_budget_bytes: Some(16 * 1024), // tiny on purpose
        ..Default::default()
    };
    let model = load_model_with(fixture_dir(), &opts).unwrap();
    let eff = undertow_core::Model::max_context(&model);
    assert!(
        eff < model.cfg.max_position_embeddings,
        "budget must shrink the window ({eff})"
    );
    let long: Vec<usize> = (0..eff + 1).map(|i| i % model.cfg.vocab_size).collect();
    match model.forward(&long) {
        Err(undertow_core::EngineError::ContextOverflow { max, .. }) => assert_eq!(max, eff),
        other => panic!("expected overflow at the lowered bound, got {other:?}"),
    }
    // Within the lowered bound everything works.
    assert!(model.forward(&long[..eff.min(8)]).is_ok());
}
