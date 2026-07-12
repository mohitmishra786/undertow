//! Integration tests against a live server over the oracle fixture.
//!
//! The oracle ships a byte-level tokenizer (every byte = one token id
//! within its 256-token vocab), so the full HTTP → template → tokenize →
//! generate → detokenize path runs hermetically, no downloads.

use std::path::PathBuf;
use std::sync::Arc;

use engine_core::model::{LoadOptions, StoreChoice};
use engine_server::{router, ServerState};

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../engine-bench/fixtures/oracle-tiny")
}

async fn spawn_server() -> String {
    let opts = LoadOptions {
        store: StoreChoice::Resident,
        ..Default::default()
    };
    let model = deepseek_moe::loader::load_model_with(fixture_dir(), &opts).unwrap();
    let tokenizer = engine_tokenizer::Tokenizer::from_dir(fixture_dir()).unwrap();
    let state = ServerState::new(Box::new(model), tokenizer, "oracle-tiny".into());
    let app = router(Arc::new(state));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    format!("http://{addr}")
}

#[tokio::test(flavor = "multi_thread")]
async fn health_and_models() {
    let base = spawn_server().await;
    let client = reqwest::Client::new();

    let health: serde_json::Value = client
        .get(format!("{base}/health"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(health["status"], "ok");

    let models: serde_json::Value = client
        .get(format!("{base}/v1/models"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(models["data"][0]["id"], "oracle-tiny");
}

#[tokio::test(flavor = "multi_thread")]
async fn chat_completion_unary_shape_and_determinism() {
    let base = spawn_server().await;
    let client = reqwest::Client::new();
    let req = serde_json::json!({
        "model": "oracle-tiny",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 12,
        "temperature": 0,
    });
    let a: serde_json::Value = client
        .post(format!("{base}/v1/chat/completions"))
        .json(&req)
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(a["object"], "chat.completion");
    assert_eq!(a["choices"][0]["message"]["role"], "assistant");
    assert_eq!(a["usage"]["completion_tokens"], 12);
    assert_eq!(a["choices"][0]["finish_reason"], "length");
    let text_a = a["choices"][0]["message"]["content"].as_str().unwrap();

    // Greedy is deterministic: same request, same bytes.
    let b: serde_json::Value = client
        .post(format!("{base}/v1/chat/completions"))
        .json(&req)
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(
        text_a,
        b["choices"][0]["message"]["content"].as_str().unwrap()
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn chat_completion_streaming_matches_unary() {
    let base = spawn_server().await;
    let client = reqwest::Client::new();
    let mk = |stream: bool| {
        serde_json::json!({
            "messages": [{"role": "user", "content": "abc"}],
            "max_tokens": 10,
            "temperature": 0,
            "stream": stream,
        })
    };
    let unary: serde_json::Value = client
        .post(format!("{base}/v1/chat/completions"))
        .json(&mk(false))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let expect = unary["choices"][0]["message"]["content"].as_str().unwrap();

    let body = client
        .post(format!("{base}/v1/chat/completions"))
        .json(&mk(true))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    let mut streamed = String::new();
    let mut saw_done = false;
    let mut finish: Option<String> = None;
    for line in body.lines() {
        let Some(data) = line.strip_prefix("data: ") else {
            continue;
        };
        if data == "[DONE]" {
            saw_done = true;
            continue;
        }
        let chunk: serde_json::Value = serde_json::from_str(data).unwrap();
        assert_eq!(chunk["object"], "chat.completion.chunk");
        if let Some(t) = chunk["choices"][0]["delta"]["content"].as_str() {
            streamed.push_str(t);
        }
        if let Some(f) = chunk["choices"][0]["finish_reason"].as_str() {
            finish = Some(f.to_string());
        }
    }
    assert!(saw_done, "stream must end with [DONE]");
    assert_eq!(finish.as_deref(), Some("length"));
    assert_eq!(streamed, expect, "streamed text differs from unary");
}

#[tokio::test(flavor = "multi_thread")]
async fn completions_endpoint_and_stop_strings() {
    let base = spawn_server().await;
    let client = reqwest::Client::new();
    let full: serde_json::Value = client
        .post(format!("{base}/v1/completions"))
        .json(&serde_json::json!({
            "prompt": "hello", "max_tokens": 16, "temperature": 0,
        }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(full["object"], "text_completion");
    let text = full["choices"][0]["text"].as_str().unwrap().to_string();
    assert!(!text.is_empty());

    // Use a substring of the deterministic output as a stop string: the
    // response must be cut right before it.
    let chars: Vec<char> = text.chars().collect();
    if chars.len() >= 4 {
        let stop: String = chars[2..4.min(chars.len())].iter().collect();
        let stopped: serde_json::Value = client
            .post(format!("{base}/v1/completions"))
            .json(&serde_json::json!({
                "prompt": "hello", "max_tokens": 16, "temperature": 0, "stop": stop,
            }))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        let cut = stopped["choices"][0]["text"].as_str().unwrap();
        assert!(
            !cut.contains(stopped["choices"][0]["text"].as_str().unwrap())
                || !cut.contains(&stop.to_string()),
            "stop string leaked into output: {cut:?}"
        );
        assert!(text.starts_with(cut), "stopped output must be a prefix");
        assert_eq!(stopped["choices"][0]["finish_reason"], "stop");
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn bad_requests_are_4xx() {
    let base = spawn_server().await;
    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{base}/v1/chat/completions"))
        .json(&serde_json::json!({ "messages": [] }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 400);

    let resp = client
        .post(format!("{base}/v1/chat/completions"))
        .body("not json")
        .header("content-type", "application/json")
        .send()
        .await
        .unwrap();
    assert!(resp.status().is_client_error());
}

#[tokio::test(flavor = "multi_thread")]
async fn concurrent_requests_serialize_and_all_answer() {
    let base = spawn_server().await;
    let client = reqwest::Client::new();
    let mut handles = Vec::new();
    for i in 0..4 {
        let client = client.clone();
        let base = base.clone();
        handles.push(tokio::spawn(async move {
            let v: serde_json::Value = client
                .post(format!("{base}/v1/chat/completions"))
                .json(&serde_json::json!({
                    "messages": [{"role": "user", "content": format!("msg {i}")}],
                    "max_tokens": 6,
                    "temperature": 0,
                }))
                .send()
                .await
                .unwrap()
                .json()
                .await
                .unwrap();
            assert_eq!(v["usage"]["completion_tokens"], 6);
        }));
    }
    for h in handles {
        h.await.unwrap();
    }
}
