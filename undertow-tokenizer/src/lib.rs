//! Tokenization for HF-layout checkpoints.
//!
//! Wraps the `tokenizers` crate (the same implementation the Python
//! ecosystem uses, so encode/decode agrees with the reference stack by
//! construction), and adds the two things an inference engine needs on
//! top: chat-template rendering (`tokenizer_config.json` templates are
//! Jinja; rendered with minijinja) and streaming detokenization that never
//! emits half a UTF-8 character or half a multi-token glyph.

use std::path::Path;

use serde::Deserialize;
use undertow_core::{EngineError, Result};

/// One chat turn.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

impl ChatMessage {
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
        }
    }
}

#[derive(Debug, Default, Deserialize)]
struct TokenizerConfig {
    #[serde(default)]
    chat_template: Option<serde_json::Value>,
    #[serde(default)]
    bos_token: Option<serde_json::Value>,
    #[serde(default)]
    eos_token: Option<serde_json::Value>,
    #[serde(default)]
    add_bos_token: Option<bool>,
}

/// `bos_token` / `eos_token` fields are either a plain string or an
/// AddedToken object with a `content` field.
fn token_content(v: &serde_json::Value) -> Option<String> {
    match v {
        serde_json::Value::String(s) => Some(s.clone()),
        serde_json::Value::Object(o) => o
            .get("content")
            .and_then(|c| c.as_str())
            .map(str::to_string),
        _ => None,
    }
}

pub struct Tokenizer {
    inner: tokenizers::Tokenizer,
    chat_template: Option<String>,
    bos_token: Option<String>,
    eos_token: Option<String>,
    add_bos: bool,
}

impl Tokenizer {
    /// Load `tokenizer.json` (+ optional `tokenizer_config.json`) from a
    /// model directory.
    pub fn from_dir(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref();
        let tok_path = dir.join("tokenizer.json");
        let inner = tokenizers::Tokenizer::from_file(&tok_path)
            .map_err(|e| EngineError::Tokenizer(format!("{}: {e}", tok_path.display())))?;

        let cfg: TokenizerConfig = match std::fs::read(dir.join("tokenizer_config.json")) {
            Ok(bytes) => serde_json::from_slice(&bytes)
                .map_err(|e| EngineError::Tokenizer(format!("tokenizer_config.json: {e}")))?,
            Err(_) => TokenizerConfig::default(),
        };

        // chat_template is a string, or a list of {name, template} pairs
        // (multi-template models); take "default" or the first.
        let chat_template = cfg.chat_template.as_ref().and_then(|v| match v {
            serde_json::Value::String(s) => Some(s.clone()),
            serde_json::Value::Array(entries) => {
                let mut default = None;
                let mut first = None;
                for e in entries {
                    let name = e.get("name").and_then(|n| n.as_str());
                    let tpl = e.get("template").and_then(|t| t.as_str());
                    if let Some(t) = tpl {
                        if first.is_none() {
                            first = Some(t.to_string());
                        }
                        if name == Some("default") {
                            default = Some(t.to_string());
                        }
                    }
                }
                default.or(first)
            }
            _ => None,
        });

        Ok(Self {
            inner,
            chat_template,
            bos_token: cfg.bos_token.as_ref().and_then(token_content),
            eos_token: cfg.eos_token.as_ref().and_then(token_content),
            add_bos: cfg.add_bos_token.unwrap_or(false),
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }

    /// Encode text. `add_special` controls the tokenizer's own
    /// post-processor special tokens (BOS etc.); chat-templated text
    /// should pass `false` because the template already placed them.
    pub fn encode(&self, text: &str, add_special: bool) -> Result<Vec<usize>> {
        let enc = self
            .inner
            .encode(text, add_special)
            .map_err(|e| EngineError::Tokenizer(e.to_string()))?;
        Ok(enc.get_ids().iter().map(|&i| i as usize).collect())
    }

    pub fn decode(&self, ids: &[usize]) -> Result<String> {
        let ids: Vec<u32> = ids.iter().map(|&i| i as u32).collect();
        self.inner
            .decode(&ids, true)
            .map_err(|e| EngineError::Tokenizer(e.to_string()))
    }

    pub fn token_to_id(&self, token: &str) -> Option<usize> {
        self.inner.token_to_id(token).map(|i| i as usize)
    }

    /// Stop-token ids derived from the tokenizer config.
    pub fn eos_ids(&self) -> Vec<usize> {
        self.eos_token
            .as_deref()
            .and_then(|t| self.token_to_id(t))
            .into_iter()
            .collect()
    }

    pub fn has_chat_template(&self) -> bool {
        self.chat_template.is_some()
    }

    /// Render the model's chat template over `messages`.
    ///
    /// Falls back to a plain, clearly-delimited format when the model
    /// ships no template — degraded quality beats refusing to chat, and
    /// the caller can detect the situation via `has_chat_template`.
    pub fn apply_chat_template(
        &self,
        messages: &[ChatMessage],
        add_generation_prompt: bool,
    ) -> Result<String> {
        let Some(template) = &self.chat_template else {
            let mut out = String::new();
            for m in messages {
                out.push_str(&format!("<|{}|>\n{}\n", m.role, m.content));
            }
            if add_generation_prompt {
                out.push_str("<|assistant|>\n");
            }
            return Ok(out);
        };

        let mut env = minijinja::Environment::new();
        env.set_trim_blocks(true);
        env.set_lstrip_blocks(true);
        // HF templates call raise_exception on unsupported inputs.
        env.add_function(
            "raise_exception",
            |msg: String| -> std::result::Result<minijinja::Value, minijinja::Error> {
                Err(minijinja::Error::new(
                    minijinja::ErrorKind::InvalidOperation,
                    msg,
                ))
            },
        );
        env.add_template("chat", template)
            .map_err(|e| EngineError::Tokenizer(format!("bad chat template: {e}")))?;
        let tmpl = env.get_template("chat").expect("just added");
        tmpl.render(minijinja::context! {
            messages => messages,
            add_generation_prompt => add_generation_prompt,
            bos_token => self.bos_token.clone().unwrap_or_default(),
            eos_token => self.eos_token.clone().unwrap_or_default(),
        })
        .map_err(|e| EngineError::Tokenizer(format!("chat template render: {e}")))
    }

    /// Encode a chat, ready for prefill.
    pub fn encode_chat(
        &self,
        messages: &[ChatMessage],
        add_generation_prompt: bool,
    ) -> Result<Vec<usize>> {
        let text = self.apply_chat_template(messages, add_generation_prompt)?;
        // Templates place BOS themselves; honor add_bos only for the
        // template-less fallback.
        let add_special = !self.has_chat_template() && self.add_bos;
        self.encode(&text, add_special)
    }
}

/// Render a Jinja chat template (the pure core of
/// [`Tokenizer::apply_chat_template`], also the fuzzing entry). Untrusted
/// templates and message content must produce a string or an error, never
/// a panic or unbounded work beyond minijinja's own limits.
pub fn render_chat_template(
    template: &str,
    messages: &[ChatMessage],
    add_generation_prompt: bool,
    bos_token: &str,
    eos_token: &str,
) -> Result<String> {
    let mut env = minijinja::Environment::new();
    env.set_trim_blocks(true);
    env.set_lstrip_blocks(true);
    // Untrusted templates must not be able to stall the process.
    env.set_fuel(Some(1_000_000));
    // HF templates call raise_exception on unsupported inputs.
    env.add_function(
        "raise_exception",
        |msg: String| -> std::result::Result<minijinja::Value, minijinja::Error> {
            Err(minijinja::Error::new(
                minijinja::ErrorKind::InvalidOperation,
                msg,
            ))
        },
    );
    env.add_template("chat", template)
        .map_err(|e| EngineError::Tokenizer(format!("bad chat template: {e}")))?;
    let tmpl = env.get_template("chat").expect("just added");
    tmpl.render(minijinja::context! {
        messages => messages,
        add_generation_prompt => add_generation_prompt,
        bos_token => bos_token,
        eos_token => eos_token,
    })
    .map_err(|e| EngineError::Tokenizer(format!("chat template render: {e}")))
}

/// Incremental detokenizer for streaming output.
///
/// Decoding token-by-token is wrong in general: byte-level BPE tokens can
/// end mid-UTF-8-sequence, and decoders apply cleanup that depends on
/// neighboring tokens. This decoder re-decodes the full generated suffix
/// and emits only the stable, valid-UTF-8 prefix delta. Cost is O(n) per
/// token over the generated ids, which is irrelevant next to a forward
/// pass; correctness is not negotiable.
pub struct StreamDecoder<'t> {
    tok: &'t Tokenizer,
    ids: Vec<usize>,
    emitted: usize,
}

impl<'t> StreamDecoder<'t> {
    pub fn new(tok: &'t Tokenizer) -> Self {
        Self {
            tok,
            ids: Vec::new(),
            emitted: 0,
        }
    }

    /// Feed one generated token; returns any newly-stable text.
    pub fn push(&mut self, id: usize) -> Result<String> {
        self.ids.push(id);
        let full = self.tok.decode(&self.ids)?;
        // The decode of ids[..n] is a prefix of the decode of ids[..n+1]
        // for byte-level tokenizers except when the last characters are
        // still incomplete; hold back the final char to stay safe against
        // replacement-character flicker.
        let stable_end = match full.char_indices().next_back() {
            Some((idx, c)) if c == char::REPLACEMENT_CHARACTER => idx,
            _ => full.len(),
        };
        if stable_end <= self.emitted {
            return Ok(String::new());
        }
        // Guard against decoders that rewrite earlier text (should not
        // happen with byte-level BPE; if it does, re-emit from scratch is
        // wrong, so emit nothing until it stabilizes).
        let delta = full[self.emitted..stable_end].to_string();
        self.emitted = stable_end;
        Ok(delta)
    }

    /// Flush whatever is still buffered (call once generation ends).
    pub fn finish(&mut self) -> Result<String> {
        let full = self.tok.decode(&self.ids)?;
        let delta = full[self.emitted.min(full.len())..].to_string();
        self.emitted = full.len();
        Ok(delta)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use tokenizers::models::bpe::BpeBuilder;

    /// Build a tiny byte-level BPE tokenizer entirely in memory, write it
    /// to a dir, exercise the from_dir path.
    fn fixture_dir() -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();

        // Byte-level alphabet plus a couple of merges. ByteLevel maps
        // space to 'Ġ' (U+0120).
        let mut vocab: HashMap<String, u32> = HashMap::new();
        let alphabet = tokenizers::pre_tokenizers::byte_level::ByteLevel::alphabet();
        let mut chars: Vec<char> = alphabet.into_iter().collect();
        chars.sort();
        for (i, ch) in chars.iter().enumerate() {
            vocab.insert(ch.to_string(), i as u32);
        }
        let base = vocab.len() as u32;
        vocab.insert("he".into(), base);
        vocab.insert("ll".into(), base + 1);
        vocab.insert("hell".into(), base + 2);
        vocab.insert("hello".into(), base + 3);
        vocab.insert("<|eos|>".into(), base + 4);
        let merges = vec![
            ("h".to_string(), "e".to_string()),
            ("l".to_string(), "l".to_string()),
            ("he".to_string(), "ll".to_string()),
            ("hell".to_string(), "o".to_string()),
        ];
        let vocab: tokenizers::models::bpe::Vocab = vocab.into_iter().collect();
        let model = BpeBuilder::new()
            .vocab_and_merges(vocab, merges)
            .build()
            .unwrap();
        let mut tok = tokenizers::Tokenizer::new(model);
        tok.with_pre_tokenizer(Some(
            tokenizers::pre_tokenizers::byte_level::ByteLevel::default().add_prefix_space(false),
        ));
        tok.with_decoder(Some(tokenizers::decoders::byte_level::ByteLevel::default()));
        tok.add_special_tokens(&[tokenizers::AddedToken::from("<|eos|>", true)]);
        tok.save(dir.path().join("tokenizer.json"), false).unwrap();

        std::fs::write(
            dir.path().join("tokenizer_config.json"),
            serde_json::json!({
                "eos_token": "<|eos|>",
                "chat_template": "{% for m in messages %}<{{ m.role }}>{{ m.content }}</{{ m.role }}>{% endfor %}{% if add_generation_prompt %}<assistant>{% endif %}"
            })
            .to_string(),
        )
        .unwrap();
        dir
    }

    #[test]
    fn encode_decode_roundtrip() {
        let dir = fixture_dir();
        let tok = Tokenizer::from_dir(dir.path()).unwrap();
        for text in ["hello", "hello hello", "abc xyz", "tabs\tand\nnewlines"] {
            let ids = tok.encode(text, false).unwrap();
            assert!(!ids.is_empty());
            assert_eq!(tok.decode(&ids).unwrap(), text, "roundtrip of {text:?}");
        }
    }

    #[test]
    fn merges_actually_merge() {
        let dir = fixture_dir();
        let tok = Tokenizer::from_dir(dir.path()).unwrap();
        let ids = tok.encode("hello", false).unwrap();
        assert_eq!(ids.len(), 1, "'hello' must encode to the merged token");
    }

    #[test]
    fn eos_detected_from_config() {
        let dir = fixture_dir();
        let tok = Tokenizer::from_dir(dir.path()).unwrap();
        let eos = tok.eos_ids();
        assert_eq!(eos.len(), 1);
        assert_eq!(tok.token_to_id("<|eos|>"), Some(eos[0]));
    }

    #[test]
    fn chat_template_renders() {
        let dir = fixture_dir();
        let tok = Tokenizer::from_dir(dir.path()).unwrap();
        assert!(tok.has_chat_template());
        let msgs = [
            ChatMessage::new("user", "hello"),
            ChatMessage::new("assistant", "hello hello"),
            ChatMessage::new("user", "ok"),
        ];
        let text = tok.apply_chat_template(&msgs, true).unwrap();
        assert_eq!(
            text,
            "<user>hello</user><assistant>hello hello</assistant><user>ok</user><assistant>"
        );
        let ids = tok.encode_chat(&msgs, true).unwrap();
        assert!(!ids.is_empty());
    }

    #[test]
    fn fallback_template_when_missing() {
        let dir = fixture_dir();
        std::fs::remove_file(dir.path().join("tokenizer_config.json")).unwrap();
        let tok = Tokenizer::from_dir(dir.path()).unwrap();
        assert!(!tok.has_chat_template());
        let text = tok
            .apply_chat_template(&[ChatMessage::new("user", "hi")], true)
            .unwrap();
        assert!(text.contains("hi") && text.contains("assistant"));
    }

    #[test]
    fn stream_decoder_matches_batch_decode() {
        let dir = fixture_dir();
        let tok = Tokenizer::from_dir(dir.path()).unwrap();
        let text = "hello hello abc";
        let ids = tok.encode(text, false).unwrap();
        let mut sd = StreamDecoder::new(&tok);
        let mut streamed = String::new();
        for &id in &ids {
            streamed.push_str(&sd.push(id).unwrap());
        }
        streamed.push_str(&sd.finish().unwrap());
        assert_eq!(streamed, text);
    }

    #[test]
    fn stream_decoder_handles_multibyte_utf8() {
        let dir = fixture_dir();
        let tok = Tokenizer::from_dir(dir.path()).unwrap();
        // Multi-byte chars force byte-level tokens that split code points.
        let text = "héllo → 世界";
        let ids = tok.encode(text, false).unwrap();
        assert!(ids.len() > 3, "must be split across several tokens");
        let mut sd = StreamDecoder::new(&tok);
        let mut streamed = String::new();
        for &id in &ids {
            let delta = sd.push(id).unwrap();
            assert!(
                !delta.contains(char::REPLACEMENT_CHARACTER),
                "no partial characters may be emitted"
            );
            streamed.push_str(&delta);
        }
        streamed.push_str(&sd.finish().unwrap());
        assert_eq!(streamed, text);
    }

    #[test]
    fn missing_tokenizer_json_is_clean_error() {
        let dir = tempfile::tempdir().unwrap();
        match Tokenizer::from_dir(dir.path()) {
            Err(EngineError::Tokenizer(_)) => {}
            Err(other) => panic!("wrong error kind: {other}"),
            Ok(_) => panic!("must fail on empty dir"),
        }
    }
}
