// src/storage/mod.rs

pub mod segment;

/// Re-exporting transactional memory segment partitions and query result mappings.
pub use segment::{SearchResult, Segment};
