//! safetensors I/O with positioned reads (`pread`), never `mmap`.
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
mod writer;

pub use reader::{Dtype, SafetensorsReader, ShardedModelReader, TensorInfo};
pub use writer::write_safetensors;
