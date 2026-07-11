//! Tier scheduler — placeholder for Phase 1+.
//!
//! Will own the interplay between routing decisions, cache state, and
//! speculative prefetch (MoE-SpeQ-style draft routing): deciding *when* to
//! issue disk reads so I/O overlaps expert matmuls instead of serializing
//! behind them.
//!
//! Kept as a named module from day one so the trait boundaries around it
//! (`TieredStore::prefetch`, `ExpertCache::record_use`) are designed with a
//! consumer in mind, but deliberately not implemented until the scalar
//! reference forward pass is validated.

/// Placeholder. Phase 1 gives this a real API.
pub struct Scheduler;
