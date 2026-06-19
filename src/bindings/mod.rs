// src/bindings/mod.rs

pub mod payload;
pub mod storage;

/// Root-level re-export of the Python-exposed VectorStorage class.
/// This acts as the primary FFI boundary orchestration layer.
pub use storage::PyVectorStorage;
