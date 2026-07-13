//! config.json is untrusted input; every family's parse+validate must
//! reject garbage without panicking (including division-shaped checks
//! like head geometry and group counts).

#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let _ = undertow_deepseek_moe::DeepseekConfig::from_slice(data);
});
