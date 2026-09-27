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

use std::io::{BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use undertow_convert::{convert_hf_with_resolver, HfConfig};

struct MockHfServer {
    addr: String,
    running: Arc<AtomicBool>,
    handle: Option<thread::JoinHandle<()>>,
}

impl MockHfServer {
    fn start(fixture_dir: PathBuf) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let addr = format!("http://127.0.0.1:{port}");
        let running = Arc::new(AtomicBool::new(true));
        let running_clone = running.clone();

        listener.set_nonblocking(true).unwrap();

        let handle = thread::spawn(move || {
            while running_clone.load(Ordering::Relaxed) {
                match listener.accept() {
                    Ok((stream, _)) => {
                        let _ = stream.set_nonblocking(false);
                        Self::handle_connection(stream, &fixture_dir);
                    }
                    Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                        thread::sleep(std::time::Duration::from_millis(5));
                    }
                    Err(_) => break,
                }
            }
        });

        Self {
            addr,
            running,
            handle: Some(handle),
        }
    }

    fn handle_connection(mut stream: TcpStream, fixture: &Path) {
        let mut reader = BufReader::new(stream.try_clone().unwrap());
        let mut request_line = String::new();
        if reader.read_line(&mut request_line).unwrap_or(0) == 0 {
            return;
        }
        let parts: Vec<&str> = request_line.split_whitespace().collect();
        if parts.len() < 2 {
            return;
        }
        let method = parts[0];
        let path = parts[1];

        let mut range_header = None;
        let mut line = String::new();
        while reader.read_line(&mut line).unwrap_or(0) > 0 {
            if line == "\r\n" || line == "\n" {
                break;
            }
            if line.to_ascii_lowercase().starts_with("range:") {
                let val = line.split(':').nth(1).unwrap_or("").trim().to_string();
                range_header = Some(val);
            }
            line.clear();
        }

        let filename = path.split('/').next_back().unwrap_or("");
        let file_path = fixture.join(filename);
        if !file_path.exists() {
            let response =
                "HTTP/1.1 404 Not Found\r\nConnection: close\r\nContent-Length: 0\r\n\r\n";
            let _ = stream.write_all(response.as_bytes());
            let _ = stream.flush();
            return;
        }

        let content = std::fs::read(&file_path).unwrap();
        let total = content.len();

        if method == "HEAD" {
            let response = format!(
                "HTTP/1.1 200 OK\r\nConnection: close\r\nAccept-Ranges: bytes\r\nContent-Length: {total}\r\n\r\n"
            );
            let _ = stream.write_all(response.as_bytes());
            let _ = stream.flush();
            return;
        }

        if let Some(range) = range_header {
            if let Some(bytes_range) = range.strip_prefix("bytes=") {
                let range_parts: Vec<&str> = bytes_range.split('-').collect();
                if range_parts.len() == 2 {
                    let start: usize = range_parts[0].parse().unwrap_or(0);
                    let end: usize = range_parts[1].parse().unwrap_or(total.saturating_sub(1));
                    let end = end.min(total.saturating_sub(1));
                    if start <= end && start < total {
                        let slice = &content[start..=end];
                        let len = slice.len();
                        let header = format!(
                            "HTTP/1.1 206 Partial Content\r\nConnection: close\r\nContent-Range: bytes {start}-{end}/{total}\r\nContent-Length: {len}\r\n\r\n"
                        );
                        let _ = stream.write_all(header.as_bytes());
                        let _ = stream.write_all(slice);
                        let _ = stream.flush();
                        return;
                    }
                }
            }
        }

        let header = format!(
            "HTTP/1.1 200 OK\r\nConnection: close\r\nAccept-Ranges: bytes\r\nContent-Length: {total}\r\n\r\n"
        );
        let _ = stream.write_all(header.as_bytes());
        let _ = stream.write_all(&content);
        let _ = stream.flush();
    }
}

impl Drop for MockHfServer {
    fn drop(&mut self) {
        self.running.store(false, Ordering::Relaxed);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

#[test]
fn convert_hf_streaming_matches_local() {
    let fixture = fixture_dir();
    let server = MockHfServer::start(fixture.clone());

    let local_dst = tempdir().unwrap();
    let remote_dst = tempdir().unwrap();

    let opts = ConvertOptions {
        row_chunk: 32,
        force: true,
        ..Default::default()
    };

    // 1. Convert locally
    let local_report = convert(
        &fixture,
        local_dst.path(),
        &undertow_deepseek_moe::classify_tensor,
        &opts,
    )
    .unwrap();

    // 2. Convert via remote HF streaming
    let hf_config = HfConfig::new("test-org/oracle-tiny").with_endpoint(server.addr.clone());
    let remote_report = convert_hf_with_resolver(
        hf_config.clone(),
        remote_dst.path(),
        |_| Ok(Box::new(undertow_deepseek_moe::classify_tensor)),
        &opts,
    )
    .unwrap();

    assert_eq!(local_report.tensors, remote_report.tensors);
    assert_eq!(local_report.shards_written, remote_report.shards_written);
    assert_eq!(local_report.bytes_out, remote_report.bytes_out);

    // Verify all generated safetensors match bit-for-bit
    for entry in std::fs::read_dir(local_dst.path()).unwrap() {
        let entry = entry.unwrap();
        let fname = entry.file_name().into_string().unwrap();
        if fname.ends_with(".safetensors") {
            let local_bytes = std::fs::read(entry.path()).unwrap();
            let remote_bytes = std::fs::read(remote_dst.path().join(&fname))
                .unwrap_or_else(|e| panic!("missing remote shard {fname}: {e}"));
            assert_eq!(
                local_bytes, remote_bytes,
                "shard {fname} differs between local and remote streaming conversion"
            );
        }
    }

    // 3. Resumability: running again on remote_dst without force should skip all shards
    let resume_opts = ConvertOptions {
        force: false,
        ..opts
    };
    let resume_report = convert_hf_with_resolver(
        hf_config,
        remote_dst.path(),
        |_| Ok(Box::new(undertow_deepseek_moe::classify_tensor)),
        &resume_opts,
    )
    .unwrap();

    assert_eq!(resume_report.shards_written, 0);
    assert_eq!(resume_report.shards_skipped, remote_report.shards_written);
}

#[test]
fn convert_hf_non_finite_rejected() {
    let bad_src = tempdir().unwrap();
    let bad_tensor = undertow_core::Tensor::new(vec![2, 2], vec![1.0, f32::NAN, 0.0, 2.0]);
    undertow_io::write_safetensors(
        bad_src.path().join("model.safetensors"),
        &[("dense.w".into(), &bad_tensor)],
    )
    .unwrap();
    std::fs::write(
        bad_src.path().join("config.json"),
        r#"{"model_type": "deepseek_v3"}"#,
    )
    .unwrap();

    let server = MockHfServer::start(bad_src.path().to_path_buf());
    let remote_dst = tempdir().unwrap();
    let hf_config = HfConfig::new("test-org/bad-model").with_endpoint(server.addr.clone());

    let err = convert_hf_with_resolver(
        hf_config,
        remote_dst.path(),
        |_| Ok(Box::new(undertow_deepseek_moe::classify_tensor)),
        &ConvertOptions::default(),
    )
    .unwrap_err();

    assert!(err.to_string().contains("non-finite"), "{err}");
}
