//! Physical-memory detection for cache auto-sizing.
//!
//! The store's cache budget defaults to a fraction of physical RAM minus
//! the resident dense weights, never overcommitted, always overridable by
//! an explicit flag. On Linux we can additionally read `MemAvailable`,
//! which reflects reclaimable page cache; macOS has no equally honest
//! single number, so the total-based heuristic applies there.

/// Total physical memory in bytes, if detectable on this platform.
pub fn total_memory_bytes() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        parse_meminfo_kib("MemTotal:").map(|kib| kib * 1024)
    }
    #[cfg(target_os = "macos")]
    {
        let mut size: u64 = 0;
        let mut len = std::mem::size_of::<u64>();
        let name = std::ffi::CString::new("hw.memsize").expect("static name");
        // SAFETY: sysctlbyname with a properly sized output buffer for the
        // documented u64 hw.memsize key.
        let rc = unsafe {
            libc::sysctlbyname(
                name.as_ptr(),
                &mut size as *mut u64 as *mut libc::c_void,
                &mut len,
                std::ptr::null_mut(),
                0,
            )
        };
        if rc == 0 && size > 0 {
            Some(size)
        } else {
            None
        }
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        None
    }
}

/// Memory the OS reports as available right now (Linux `MemAvailable`).
/// `None` where the platform has no equivalent; callers fall back to the
/// total-based heuristic.
pub fn available_memory_bytes() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        parse_meminfo_kib("MemAvailable:").map(|kib| kib * 1024)
    }
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

#[cfg(target_os = "linux")]
fn parse_meminfo_kib(key: &str) -> Option<u64> {
    let text = std::fs::read_to_string("/proc/meminfo").ok()?;
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix(key) {
            return rest.trim().trim_end_matches(" kB").trim().parse().ok();
        }
    }
    None
}

/// Cache budget when the user did not pass one explicitly.
///
/// Policy: half of what the OS calls available (or half of total when
/// availability is unknown), minus the dense weights that are already
/// resident, clamped to `[64 MiB, remaining]`. Conservative on purpose:
/// an engine that OOMs the machine it runs on has failed its one job.
pub fn auto_cache_budget(dense_resident_bytes: u64) -> u64 {
    const FLOOR: u64 = 64 * 1024 * 1024;
    let base = available_memory_bytes()
        .or_else(total_memory_bytes)
        .unwrap_or(2 * 1024 * 1024 * 1024);
    let half = base / 2;
    half.saturating_sub(dense_resident_bytes).max(FLOOR)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn total_memory_detected_on_dev_platforms() {
        #[cfg(any(target_os = "linux", target_os = "macos"))]
        {
            let total = total_memory_bytes().expect("should detect RAM");
            assert!(total >= 1 << 30, "implausibly small RAM: {total}");
        }
    }

    #[test]
    fn auto_budget_has_floor() {
        assert!(auto_cache_budget(u64::MAX / 2) >= 64 * 1024 * 1024);
    }

    #[test]
    fn auto_budget_subtracts_dense() {
        let free = auto_cache_budget(0);
        let less = auto_cache_budget(32 * 1024 * 1024);
        assert!(less <= free);
    }
}
