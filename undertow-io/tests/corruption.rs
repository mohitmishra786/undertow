//! Checkpoints arrive from untrusted mirrors over unreliable transfers.
//! Every corruption mode must surface as an error (or, for silent payload
//! bit-flips, as plainly wrong-but-finite values), never a panic, hang or
//! out-of-bounds read.

use undertow_core::Tensor;
use undertow_io::{write_safetensors, SafetensorsReader};

fn make_file(dir: &std::path::Path) -> std::path::PathBuf {
    let path = dir.join("model.safetensors");
    let t = Tensor::new(vec![8, 16], (0..128).map(|i| i as f32 * 0.5).collect());
    let u = Tensor::new(vec![4], vec![1.0, 2.0, 3.0, 4.0]);
    write_safetensors(&path, &[("w".into(), &t), ("norm".into(), &u)]).unwrap();
    path
}

#[test]
fn truncated_file_is_an_error() {
    let dir = tempfile::tempdir().unwrap();
    let path = make_file(dir.path());
    let full = std::fs::read(&path).unwrap();

    // Truncation at every interesting boundary: inside the length prefix,
    // inside the header, inside the payload.
    for cut in [4usize, 12, full.len() / 2, full.len() - 3] {
        std::fs::write(&path, &full[..cut]).unwrap();
        match SafetensorsReader::open(&path) {
            Err(_) => {} // rejected at open: fine
            Ok(reader) => {
                // Header may have survived; the payload read must fail.
                assert!(
                    reader.read_f32("w").is_err() || reader.read_f32("norm").is_err(),
                    "cut at {cut}: truncated payload must not read cleanly"
                );
            }
        }
    }
}

#[test]
fn corrupt_header_length_is_an_error() {
    let dir = tempfile::tempdir().unwrap();
    let path = make_file(dir.path());
    let mut bytes = std::fs::read(&path).unwrap();

    // Absurdly large header length.
    bytes[..8].copy_from_slice(&(u64::MAX / 2).to_le_bytes());
    std::fs::write(&path, &bytes).unwrap();
    assert!(SafetensorsReader::open(&path).is_err());

    // Header length pointing into the payload (invalid json there).
    let mut bytes = std::fs::read(make_file(dir.path())).unwrap();
    let real_len = u64::from_le_bytes(bytes[..8].try_into().unwrap());
    bytes[..8].copy_from_slice(&(real_len + 40).to_le_bytes());
    std::fs::write(&path, &bytes).unwrap();
    assert!(SafetensorsReader::open(&path).is_err());
}

#[test]
fn header_json_garbage_is_an_error() {
    let dir = tempfile::tempdir().unwrap();
    let path = make_file(dir.path());
    let mut bytes = std::fs::read(&path).unwrap();
    // Flip bytes inside the header json.
    for b in &mut bytes[10..30] {
        *b ^= 0xA5;
    }
    std::fs::write(&path, &bytes).unwrap();
    assert!(SafetensorsReader::open(&path).is_err());
}

#[test]
fn header_offsets_beyond_file_are_an_error() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("model.safetensors");
    // Hand-craft a header whose tensor claims more bytes than exist.
    let header = serde_json::json!({
        "w": { "dtype": "F32", "shape": [1024, 1024], "data_offsets": [0, 4194304] }
    })
    .to_string();
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
    bytes.extend_from_slice(header.as_bytes());
    bytes.extend_from_slice(&[0u8; 64]); // far less than claimed
    std::fs::write(&path, &bytes).unwrap();
    assert!(SafetensorsReader::open(&path).is_err());
}

#[test]
fn payload_bit_flips_stay_contained() {
    let dir = tempfile::tempdir().unwrap();
    let path = make_file(dir.path());
    let mut bytes = std::fs::read(&path).unwrap();
    // Flip a byte deep in the payload: undetectable without checksums, but
    // the read must stay in bounds and produce ordinary floats or an
    // error, never UB or a panic.
    let n = bytes.len();
    bytes[n - 10] ^= 0xFF;
    std::fs::write(&path, &bytes).unwrap();
    let reader = SafetensorsReader::open(&path).unwrap();
    let w = reader.read_f32("w").unwrap();
    assert_eq!(w.data.len(), 128);
    let norm = reader.read_f32("norm").unwrap();
    assert_eq!(norm.data.len(), 4);
}

#[test]
fn shape_dtype_offset_mismatch_is_an_error() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("model.safetensors");
    // Shape says 4 f32 (16 bytes) but offsets claim 12.
    let header = serde_json::json!({
        "w": { "dtype": "F32", "shape": [4], "data_offsets": [0, 12] }
    })
    .to_string();
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
    bytes.extend_from_slice(header.as_bytes());
    bytes.extend_from_slice(&[0u8; 12]);
    std::fs::write(&path, &bytes).unwrap();
    assert!(SafetensorsReader::open(&path).is_err());
}
