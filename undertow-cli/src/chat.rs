//! Interactive chat loop with streaming output and KV-prefix reuse.
//!
//! Each turn renders the whole conversation through the model's chat
//! template, then reuses the longest common token prefix with what the
//! session has already consumed — on a disk-streaming model, re-prefilling
//! a long conversation from scratch every turn is the difference between
//! seconds and minutes.

use std::io::Write;

use anyhow::Result;
use undertow_core::sample::Sampler;
use undertow_core::Model;
use undertow_tokenizer::{ChatMessage, StreamDecoder, Tokenizer};

pub fn run_chat(
    model: &dyn Model,
    tok: &Tokenizer,
    system: Option<String>,
    max_new: usize,
    mut sampler: Sampler,
) -> Result<()> {
    let mut messages: Vec<ChatMessage> = Vec::new();
    if let Some(sys) = system {
        messages.push(ChatMessage::new("system", sys));
    }
    let mut stop_ids = model.stop_ids().to_vec();
    for id in tok.eos_ids() {
        if !stop_ids.contains(&id) {
            stop_ids.push(id);
        }
    }
    if !tok.has_chat_template() {
        eprintln!("note: model ships no chat template; using a plain fallback format");
    }
    eprintln!("undertow chat — /exit to quit, /clear to reset the conversation");

    let mut session = model.new_session();
    // Token ids the session has consumed (prompt prefixes + generations).
    let mut consumed: Vec<usize> = Vec::new();
    let stdin = std::io::stdin();
    loop {
        eprint!("> ");
        let mut line = String::new();
        if stdin.read_line(&mut line)? == 0 {
            break; // EOF
        }
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        match line {
            "/exit" | "/quit" => break,
            "/clear" => {
                messages.retain(|m| m.role == "system");
                session.truncate(0);
                consumed.clear();
                eprintln!("(conversation cleared)");
                continue;
            }
            _ => {}
        }

        messages.push(ChatMessage::new("user", line));
        let prompt_ids = tok
            .encode_chat(&messages, true)
            .map_err(anyhow::Error::from)?;

        // Reuse the longest common prefix already in the KV cache. Never
        // reuse the full prompt: prefill needs at least one position to
        // produce logits for sampling.
        let common = consumed
            .iter()
            .zip(&prompt_ids)
            .take_while(|(a, b)| a == b)
            .count()
            .min(prompt_ids.len().saturating_sub(1));
        session.truncate(common);
        consumed.truncate(common);
        let suffix = &prompt_ids[common..];

        let mut decoder = StreamDecoder::new(tok);
        let mut reply_ids: Vec<usize> = Vec::new();
        let mut out = std::io::stdout().lock();
        let t0 = std::time::Instant::now();
        let produced = undertow_core::generate(
            &mut *session,
            suffix,
            max_new,
            &mut sampler,
            &stop_ids,
            |id| {
                reply_ids.push(id);
                if stop_ids.contains(&id) {
                    return true; // counted, not printed
                }
                if let Ok(text) = decoder.push(id) {
                    let _ = out.write_all(text.as_bytes());
                    let _ = out.flush();
                }
                true
            },
        )?;
        if let Ok(rest) = decoder.finish() {
            let _ = out.write_all(rest.as_bytes());
        }
        let _ = out.write_all(b"\n");
        let _ = out.flush();
        let dt = t0.elapsed().as_secs_f64();
        eprintln!(
            "({produced} tokens, {:.2} tok/s, {} prompt tokens reused)",
            produced as f64 / dt,
            common
        );

        consumed.extend_from_slice(suffix);
        consumed.extend_from_slice(&reply_ids);
        // Stop tokens live in the KV cache but not in the visible reply.
        let visible: Vec<usize> = reply_ids
            .iter()
            .copied()
            .filter(|id| !stop_ids.contains(id))
            .collect();
        let reply_text = tok.decode(&visible).map_err(anyhow::Error::from)?;
        messages.push(ChatMessage::new("assistant", reply_text));
    }
    Ok(())
}
