//! Chat templates ship inside model repos, so they are untrusted code in
//! a sandboxed language: rendering must terminate (fuel-limited) and
//! never panic, whatever the template or message content.

#![no_main]
use undertow_tokenizer::ChatMessage;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|input: (String, String, String, bool)| {
    let (template, role, content, add_gen) = input;
    let messages = [ChatMessage::new(role, content)];
    let _ = undertow_tokenizer::render_chat_template(&template, &messages, add_gen, "<s>", "</s>");
});
