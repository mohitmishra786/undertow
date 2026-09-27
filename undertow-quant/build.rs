fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rustc-check-cfg=cfg(has_avx512)");
    // Detect compiler version to determine whether AVX-512 target features
    // (avx512bw, avx512dq, avx512vl stabilized in Rust 1.89) are supported.
    if let Some(minor) = rustc_minor_version() {
        if minor >= 89 {
            println!("cargo:rustc-cfg=has_avx512");
        }
    }
}

fn rustc_minor_version() -> Option<u32> {
    let rustc = std::env::var_os("RUSTC")?;
    let output = std::process::Command::new(rustc)
        .arg("--version")
        .output()
        .ok()?;
    let version_str = std::str::from_utf8(&output.stdout).ok()?;
    // Format: "rustc 1.88.0 (..." or "rustc 1.98.1 (..."
    let mut parts = version_str.split_whitespace();
    parts.next()?; // "rustc"
    let version = parts.next()?; // "1.88.0"
    let mut num_parts = version.split('.');
    let major = num_parts.next()?;
    if major != "1" {
        return None;
    }
    num_parts.next()?.parse::<u32>().ok()
}
