//! safetensors I/O with positioned reads (`pread`), never `mmap`, plus the
//! disk-backed expert store.
//!
//! Why no mmap: with a 350–600GB expert pool behind a small RAM budget,
//! mmap hands residency control to the page cache and makes RSS unpredictable
//! — exactly what kills memory-constrained unified-memory machines. `pread`
//! into buffers we own keeps RSS flat and lets the tier scheduler decide
//! what stays resident.
//!
//! On Unix, [`std::os::unix::fs::FileExt::read_exact_at`] compiles down to
//! `pread(2)`; it is also offset-stateless, so concurrent expert fetches
//! never contend on a shared file cursor.

mod reader;
mod store;
mod writer;

pub use reader::{
    parse_header, read_qtensor, Dtype, SafetensorsReader, ShardedModelReader, TensorInfo,
};
pub use store::{DiskExpertStore, ExpertDims};
pub use writer::{
    write_safetensors, write_safetensors_entries, PlannedEntry, ShardWriter, TensorEntry,
};
