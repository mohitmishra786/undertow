//! Headers arrive from untrusted checkpoints: arbitrary bytes plus
//! arbitrary claimed file bounds must parse or error, never panic.

#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if data.len() < 16 {
        return;
    }
    let data_start = u64::from_le_bytes(data[0..8].try_into().unwrap()) % (1 << 40);
    let file_len = u64::from_le_bytes(data[8..16].try_into().unwrap()) % (1 << 40);
    let _ = undertow_io::parse_header(&data[16..], data_start, file_len, "fuzz");
});
