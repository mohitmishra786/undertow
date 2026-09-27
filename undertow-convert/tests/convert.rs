use std::path::PathBuf;
use tempfile::tempdir;
use undertow_convert::{convert, ConvertOptions};

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../undertow-bench/fixtures/oracle-tiny")
}

#[test]
fn convert_chunk_size_invariance() {
    let src = fixture_dir();
    assert!(
        src.join("config.json").exists(),
        "fixture dir must exist at {:?}",
        src
    );

    let dir_a = tempdir().unwrap();
    let dir_b = tempdir().unwrap();
    let dir_c = tempdir().unwrap();

    let chunk_sizes = [(1, dir_a.path()), (13, dir_b.path()), (1024, dir_c.path())];

    for (chunk, dst) in chunk_sizes {
        let opts = ConvertOptions {
            row_chunk: chunk,
            force: true,
            ..Default::default()
        };
        let report = convert(&src, dst, &undertow_deepseek_moe::classify_tensor, &opts)
            .unwrap_or_else(|e| panic!("conversion failed for chunk size {chunk}: {e}"));
        assert!(report.shards_written > 0);
    }

    // Collect all safetensors and index files from dir_a
    let mut files = Vec::new();
    for entry in std::fs::read_dir(dir_a.path()).unwrap() {
        let entry = entry.unwrap();
        let name = entry.file_name().into_string().unwrap();
        if name.ends_with(".safetensors") || name.ends_with(".json") {
            files.push(name);
        }
    }
    assert!(
        !files.is_empty(),
        "expected converted output files in {:?}",
        dir_a.path()
    );

    for file in &files {
        let bytes_a = std::fs::read(dir_a.path().join(file)).unwrap();
        let bytes_b = std::fs::read(dir_b.path().join(file))
            .unwrap_or_else(|e| panic!("missing {file} in dir_b: {e}"));
        let bytes_c = std::fs::read(dir_c.path().join(file))
            .unwrap_or_else(|e| panic!("missing {file} in dir_c: {e}"));

        assert_eq!(
            bytes_a, bytes_b,
            "byte mismatch in {file} between row_chunk=1 and row_chunk=13"
        );
        assert_eq!(
            bytes_a, bytes_c,
            "byte mismatch in {file} between row_chunk=1 and row_chunk=1024"
        );
    }
}
