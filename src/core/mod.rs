// src/core/mod.rs

pub mod distance;
pub mod filter;
pub mod quantization;
pub mod vector;

/// Re-exporting foundational spatial geometry and multi-precision primitives.
pub use distance::DistanceMetric;
pub use filter::Filter;
pub use quantization::{QuantizedVector, ScalarQuantizer, StoragePrecision};
pub use vector::{Payload, PayloadValue, Point};
