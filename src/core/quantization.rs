// src/core/quantization.rs

use half::f16;
use serde::{Deserialize, Serialize};

/// Supported precision layouts for vector element storage inside the database cluster partitions.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum StoragePrecision {
    /// Strict 32-bit floating-point precision. Bypasses quantization to preserve maximum semantic recall at native hardware speeds.
    F32,
    /// 16-bit half-precision floating-point. Offers an immediate 2x RAM footprint reduction with near-zero degradation of search recall.
    F16,
    /// Localized 8-bit scalar quantized unsigned integer format. Maximizes hardware throughput, yielding up to 4x RAM and disk savings.
    F8,
}

/// Dynamic, serialized polymorphic container holding multi-precision compressed vector data matrices.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum QuantizedVector {
    /// Raw uncompressed 32-bit float layout variant.
    F32(Vec<f32>),
    /// Compressed half-precision float layout variant (executes software emulation casting on standard x86 architectures).
    F16(Vec<f16>),
    /// Asymmetric 8-bit unsigned integer layout paired alongside dedicated local min/max boundary calibration states.
    F8 { bytes: Vec<u8>, min: f32, max: f32 },
}

/// Advanced multi-precision quantizer engine orchestrating high-dimensional spatial data compression tracks.
pub struct ScalarQuantizer;

impl ScalarQuantizer {
    /// Compresses a raw 32-bit floating-point coordinate array slice into the targeted storage precision layout.
    ///
    /// When executing F8 quantization, this pipeline maps scales dynamically per vector to ensure
    /// optimal dynamic range allocation and protect the system from catastrophic recall drops.
    ///
    /// # Parameters
    /// * `vector` - Uncompressed original high-dimensional coordinate array slice to transform.
    /// * `precision` - Targeted memory layout and compression variant configuration (F32, F16, or F8).
    ///
    /// # Returns
    /// A polymorphic `QuantizedVector` variant wrapping the formatted data block partition.
    pub fn quantize(vector: &[f32], precision: StoragePrecision) -> QuantizedVector {
        match precision {
            StoragePrecision::F32 => QuantizedVector::F32(vector.to_vec()),

            StoragePrecision::F16 => {
                let f16_vectors = vector.iter().map(|&x| f16::from_f32(x)).collect();
                QuantizedVector::F16(f16_vectors)
            }

            StoragePrecision::F8 => {
                if vector.is_empty() {
                    return QuantizedVector::F8 {
                        bytes: Vec::new(),
                        min: 0.0,
                        max: 0.0,
                    };
                }

                // Scan coordinates to extract explicit mathematical boundaries for asymmetric dynamic range scaling.
                let mut min = vector[0];
                let mut max = vector[0];
                for &x in vector.iter() {
                    if x < min {
                        min = x;
                    }
                    if x > max {
                        max = x;
                    }
                }

                // Evaluate structural range parameters while injecting safety thresholds to prevent division-by-zero anomalies.
                let range = max - min;
                let denom = if range.abs() < 1e-5 { 1e-5 } else { range };

                // Execute asymmetric min-max normalizations mapping coordinates tightly into discrete 8-bit unsigned fields.
                let bytes = vector
                    .iter()
                    .map(|&x| {
                        let clamped = x.clamp(min, max);
                        let normalized = (clamped - min) / denom;
                        (normalized * 255.0).round() as u8
                    })
                    .collect();

                QuantizedVector::F8 { bytes, min, max }
            }
        }
    }

    /// Dequantizes any compressed container layout block back into an explicit 32-bit floating-point vector.
    ///
    /// This recovery decoder path is heavily utilized during strict post-filtering verification gates
    /// or geometric fallback alignment validation checkpoints.
    ///
    /// # Parameters
    /// * `q_vector` - Reference to the polymorphic quantized database entity wrapper to decode.
    ///
    /// # Returns
    /// A newly allocated native `Vec<f32>` containing rehydrated spatial feature metrics.
    pub fn dequantize(q_vector: &QuantizedVector) -> Vec<f32> {
        match q_vector {
            QuantizedVector::F32(vec) => vec.clone(),

            QuantizedVector::F16(vec) => vec.iter().map(|&x| x.to_f32()).collect(),

            QuantizedVector::F8 { bytes, min, max } => {
                let range = max - min;
                bytes
                    .iter()
                    .map(|&x| {
                        let normalized = x as f32 / 255.0;
                        min + normalized * range
                    })
                    .collect()
            }
        }
    }
}
