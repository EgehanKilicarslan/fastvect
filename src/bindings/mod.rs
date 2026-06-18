// src/bindings/mod.rs

pub mod payload;
pub mod storage;

// Re-exporting for clean internal macro discovery
pub use storage::PyVectorStorage;
