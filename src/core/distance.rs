// src/core/distance.rs

use crate::QuantizedVector;

/// Supported geometric metrics utilized for calculating spatial proximity between high-dimensional vector embeddings.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum DistanceMetric {
    /// Vector dot product metric, optimal for normalized embeddings ($A \cdot B = \sum a_i b_i$).
    DotProduct,
    /// Angular cosine similarity mapping structural orientation independently of magnitude ($\frac{A \cdot B}{\|A\| \|B\|}$).
    Cosine,
    /// Absolute straight-line distance algorithm mapping geometric variance within a Cartesian space ($\sqrt{\sum (a_i - b_i)^2}$).
    HighPrecisionEuclidean,
}

/// Dispatches high-dimensional proximity evaluation targets onto specialized underlying hardware memory alignment tracks.
///
/// This routing coordinator matches execution lanes based on the shared internal storage precision layout
/// of both vector operands, maximizing cache utilization and preventing runtime type contamination.
///
/// # Parameters
/// * `a` - Reference to the first multi-precision quantized vector container operand.
/// * `b` - Reference to the second multi-precision quantized vector container operand.
/// * `metric` - The targeted distance formula configuration to evaluate across the operands.
///
/// # Returns
/// A 32-bit floating-point similarity score calculated via the matched hardware precision track.
///
/// # Errors
/// Returns an error string if an asymmetric structural layout mismatch is detected between the two operands.
pub fn compute_distance(
    a: &QuantizedVector,
    b: &QuantizedVector,
    metric: DistanceMetric,
) -> Result<f32, String> {
    match (a, b) {
        (QuantizedVector::F32(va), QuantizedVector::F32(vb)) => {
            match metric {
                DistanceMetric::DotProduct => dot_product_f32(va, vb),
                DistanceMetric::Cosine => cosine_similarity_f32(va, vb),
                DistanceMetric::HighPrecisionEuclidean => euclidean_distance_f32(va, vb),
            }
        }
        (QuantizedVector::F16(va), QuantizedVector::F16(vb)) => {
            match metric {
                DistanceMetric::DotProduct => dot_product_f16(va, vb),
                DistanceMetric::Cosine => cosine_similarity_f16(va, vb),
                DistanceMetric::HighPrecisionEuclidean => euclidean_distance_f16(va, vb),
            }
        }
        (
            QuantizedVector::F8 { bytes: ba, min: min_a, max: max_a },
            QuantizedVector::F8 { bytes: bb, min: min_b, max: max_b }
        ) => {
            match metric {
                DistanceMetric::DotProduct => dot_product_f8(ba, *min_a, *max_a, bb, *min_b, *max_b),
                DistanceMetric::Cosine => cosine_similarity_f8(ba, *min_a, *max_a, bb, *min_b, *max_b),
                DistanceMetric::HighPrecisionEuclidean => euclidean_distance_f8(ba, *min_a, *max_a, bb, *min_b, *max_b),
            }
        }
        _ => Err("Asymmetric layout mismatch: Operands must share matching structural precision layouts.".to_string()),
    }
}

// ============================================================================================
// CORE F32 NATIVE EXECUTION PIPELINES
// ============================================================================================

/// Computes the mathematical Dot Product between two equal-length uncompressed floating-point slices.
fn dot_product_f32(a: &[f32], b: &[f32]) -> Result<f32, String> {
    if a.len() != b.len() {
        return Err(
            "Dimension mismatch: Vector lengths must be structurally identical.".to_string(),
        );
    }
    Ok(a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum())
}

/// Computes the Cosine Similarity between two uncompressed floating-point directional vectors.
fn cosine_similarity_f32(a: &[f32], b: &[f32]) -> Result<f32, String> {
    let dot = dot_product_f32(a, b)?;
    let norm_a = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm_b = b.iter().map(|&x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return Err("Zero magnitude vector error: Vector magnitude cannot be zero during cosine normalization.".to_string());
    }
    Ok(dot / (norm_a * norm_b))
}

/// Computes the standard Euclidean Distance between two uncompressed Cartesian coordinates.
fn euclidean_distance_f32(a: &[f32], b: &[f32]) -> Result<f32, String> {
    if a.len() != b.len() {
        return Err(
            "Dimension mismatch: Vector lengths must be structurally identical.".to_string(),
        );
    }
    let sum_squares: f32 = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| {
            let d = x - y;
            d * d
        })
        .sum();
    Ok(sum_squares.sqrt())
}

// ============================================================================================
// OPTIMIZED HALF-PRECISION F16 IMPLEMENTATIONS
// ============================================================================================

/// Computes the accelerated Dot Product across half-precision floating-point boundaries.
fn dot_product_f16(a: &[half::f16], b: &[half::f16]) -> Result<f32, String> {
    if a.len() != b.len() {
        return Err(
            "Dimension mismatch: Vector lengths must be structurally identical.".to_string(),
        );
    }
    Ok(a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| x.to_f32() * y.to_f32())
        .sum())
}

/// Computes the Cosine Similarity across half-precision floating-point directional vectors.
fn cosine_similarity_f16(a: &[half::f16], b: &[half::f16]) -> Result<f32, String> {
    let dot = dot_product_f16(a, b)?;
    let norm_a = a
        .iter()
        .map(|&x| {
            let f = x.to_f32();
            f * f
        })
        .sum::<f32>()
        .sqrt();
    let norm_b = b
        .iter()
        .map(|&x| {
            let f = x.to_f32();
            f * f
        })
        .sum::<f32>()
        .sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return Err("Zero magnitude vector error: Vector magnitude cannot be zero during cosine normalization.".to_string());
    }
    Ok(dot / (norm_a * norm_b))
}

/// Computes the standard Euclidean Distance between two half-precision vector structures.
fn euclidean_distance_f16(a: &[half::f16], b: &[half::f16]) -> Result<f32, String> {
    if a.len() != b.len() {
        return Err(
            "Dimension mismatch: Vector lengths must be structurally identical.".to_string(),
        );
    }
    let sum_squares: f32 = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| {
            let d = x.to_f32() - y.to_f32();
            d * d
        })
        .sum();
    Ok(sum_squares.sqrt())
}

// ============================================================================================
// INT-MATH SCALAR QUANTIZED F8 HOT PATHS
// ============================================================================================

/// Computes an accelerated integer-math dot product directly over scaled 8-bit unsigned integer blocks.
///
/// This hot path optimizes hardware utilization by bypassing dequantization overheads, accumulating raw products
/// into non-overflowing 32-bit registers before applying a unified algebraic re-scaling transform pass.
fn dot_product_f8(
    a: &[u8],
    min_a: f32,
    max_a: f32,
    b: &[u8],
    min_b: f32,
    max_b: f32,
) -> Result<f32, String> {
    if a.len() != b.len() {
        return Err(
            "Dimension mismatch: Vector lengths must be structurally identical.".to_string(),
        );
    }

    let range_a = max_a - min_a;
    let range_b = max_b - min_b;
    let len = a.len() as f32;

    // Allocate high-capacity accumulation boundaries to eliminate register overflow conditions.
    let mut sum_ab: u32 = 0;
    let mut sum_a: u32 = 0;
    let mut sum_b: u32 = 0;

    for (&x, &y) in a.iter().zip(b.iter()) {
        let xi = x as u32;
        let yi = y as u32;
        sum_ab += xi * yi;
        sum_a += xi;
        sum_b += yi;
    }

    // Execute generalized matrix de-quantization mapping via algebraic equation expansion passes.
    let factor = (range_a * range_b) / (255.0 * 255.0);
    let part1 = factor * (sum_ab as f32);
    let part2 = ((range_a * min_b) / 255.0) * (sum_a as f32);
    let part3 = ((range_b * min_a) / 255.0) * (sum_b as f32);
    let part4 = len * min_a * min_b;

    Ok(part1 + part2 + part3 + part4)
}

/// Computes the Cosine Similarity across 8-bit scalar quantized integer blocks via linear expansions.
fn cosine_similarity_f8(
    a: &[u8],
    min_a: f32,
    max_a: f32,
    b: &[u8],
    min_b: f32,
    max_b: f32,
) -> Result<f32, String> {
    let dot = dot_product_f8(a, min_a, max_a, b, min_b, max_b)?;

    // Isolated structural vector norm resolver computing exact squared lengths without dequantization pipelines.
    let calc_norm = |bytes: &[u8], min_val: f32, max_val: f32| -> f32 {
        let range = max_val - min_val;
        let mut sum_sq: u32 = 0;
        let mut sum_val: u32 = 0;
        for &x in bytes.iter() {
            let xi = x as u32;
            sum_sq += xi * xi;
            sum_val += xi;
        }
        let factor = (range * range) / (255.0 * 255.0);
        let p1 = factor * (sum_sq as f32);
        let p2 = (2.0 * range * min_val / 255.0) * (sum_val as f32);
        let p3 = (bytes.len() as f32) * min_val * min_val;
        (p1 + p2 + p3).sqrt()
    };

    let norm_a = calc_norm(a, min_a, max_a);
    let norm_b = calc_norm(b, min_b, max_b);

    if norm_a == 0.0 || norm_b == 0.0 {
        return Err("Zero magnitude vector error: Vector magnitude cannot be zero during cosine normalization.".to_string());
    }
    Ok(dot / (norm_a * norm_b))
}

/// Computes the standard Euclidean Distance between two 8-bit scalar quantized vector blocks.
fn euclidean_distance_f8(
    a: &[u8],
    min_a: f32,
    max_a: f32,
    b: &[u8],
    min_b: f32,
    max_b: f32,
) -> Result<f32, String> {
    if a.len() != b.len() {
        return Err(
            "Dimension mismatch: Vector lengths must be structurally identical.".to_string(),
        );
    }

    let range_a = max_a - min_a;
    let range_b = max_b - min_b;

    let mut sum_squares: f32 = 0.0;
    for (&x, &y) in a.iter().zip(b.iter()) {
        let va = min_a + (x as f32 / 255.0) * range_a;
        let vb = min_b + (y as f32 / 255.0) * range_b;
        let diff = va - vb;
        sum_squares += diff * diff;
    }
    Ok(sum_squares.sqrt())
}
