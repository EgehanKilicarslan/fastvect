// src/lib.rs

pub mod bindings;
pub mod core;
pub mod index;
pub mod storage;

use pyo3::prelude::*;

/// Root-level industrial API re-exports for maximum developer ergonomics.
/// This acts as a clean Facade layer over our deep domain-driven architecture.
pub use core::{DistanceMetric, Filter, Payload, PayloadValue, Point, ScalarQuantizer};
pub use index::HNSWIndex;
pub use storage::{SearchResult, Segment};

/// Python extension module definition mapping high-performance Rust subroutines directly to Python runtimes.
#[pymodule]
fn fastvect(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Registers our Qdrant-inspired modular VectorStorage class directly inside the Python module space
    m.add_class::<bindings::storage::PyVectorStorage>()?;
    Ok(())
}
