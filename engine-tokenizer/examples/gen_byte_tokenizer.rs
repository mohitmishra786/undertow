//! Generate a minimal byte-level tokenizer (256 byte tokens, no merges)
//! into a model directory. Every byte maps to one token id < 256, which
//! matches the oracle models' vocab exactly — this is what lets chat and
//! the server run against synthetic checkpoints with no downloads.
//!
//!     cargo run -p engine-tokenizer --example gen_byte_tokenizer -- <dir>

use std::collections::HashMap;

fn main() {
    let dir = std::env::args()
        .nth(1)
        .expect("usage: gen_byte_tokenizer <dir>");
    let alphabet = tokenizers::pre_tokenizers::byte_level::ByteLevel::alphabet();
    let mut chars: Vec<char> = alphabet.into_iter().collect();
    chars.sort();
    assert_eq!(chars.len(), 256);
    let vocab: HashMap<String, u32> = chars
        .iter()
        .enumerate()
        .map(|(i, c)| (c.to_string(), i as u32))
        .collect();
    let vocab: tokenizers::models::bpe::Vocab = vocab.into_iter().collect();
    let model = tokenizers::models::bpe::BpeBuilder::new()
        .vocab_and_merges(vocab, Vec::new())
        .build()
        .unwrap();
    let mut tok = tokenizers::Tokenizer::new(model);
    tok.with_pre_tokenizer(Some(
        tokenizers::pre_tokenizers::byte_level::ByteLevel::default().add_prefix_space(false),
    ));
    tok.with_decoder(Some(tokenizers::decoders::byte_level::ByteLevel::default()));
    tok.save(format!("{dir}/tokenizer.json"), false).unwrap();

    std::fs::write(
        format!("{dir}/tokenizer_config.json"),
        serde_json::json!({
            "chat_template": "{% for m in messages %}<|{{ m.role }}|>{{ m.content }}{% endfor %}{% if add_generation_prompt %}<|assistant|>{% endif %}"
        })
        .to_string(),
    )
    .unwrap();
    println!("wrote byte-level tokenizer to {dir}");
}
