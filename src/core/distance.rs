// src/core/distance.rs

use crate::QuantizedVector;
use crate::core::simd;

/// Supported geometric metrics utilized for calculating spatial proximity between high-dimensional vector embeddings.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum DistanceMetric {
    /// Vector dot product metric, optimal for normalized embeddings.
    /// Matches the structural algebraic formula: $A \cdot B = \sum a_i b_i$.
    DotProduct,
    /// Angular cosine similarity mapping structural orientation independently of magnitude.
    /// Matches the structural algebraic formula: $\frac{A \cdot B}{\|A\| \|B\|}$.
    Cosine,
    /// Absolute straight-line distance algorithm mapping geometric variance within a Cartesian topology.
    /// Matches the structural algebraic formula: $\sqrt{\sum (a_i - b_i)^2}$.
    HighPrecisionEuclidean,
}

/// Dispatches high-dimensional proximity evaluation targets onto specialized underlying hardware memory alignment tracks.
///
/// Evaluates variant multi-precision data formats dynamically and routes computations through direct architecture execution paths.
///
/// # Parameters
/// * `a` - Read reference linking to the initial quantized data array container.
/// * `b` - Read reference linking to the secondary target quantized data array container.
/// * `metric` - Target spatial calculation formula token to run.
///
/// # Errors
/// Returns an explicit error string slice if type precisions reveal asymmetric structural mismatches.
pub fn compute_distance(
    a: &QuantizedVector,
    b: &QuantizedVector,
    metric: DistanceMetric,
) -> Result<f32, String> {
    match (a, b) {
        (QuantizedVector::F32(va), QuantizedVector::F32(vb)) => match metric {
            DistanceMetric::DotProduct => dot_product_f32(va, vb),
            DistanceMetric::Cosine => cosine_similarity_f32(va, vb),
            DistanceMetric::HighPrecisionEuclidean => euclidean_distance_f32(va, vb),
        },
        (QuantizedVector::F16(va), QuantizedVector::F16(vb)) => match metric {
            DistanceMetric::DotProduct => dot_product_f16(va, vb),
            DistanceMetric::Cosine => cosine_similarity_f16(va, vb),
            DistanceMetric::HighPrecisionEuclidean => euclidean_distance_f16(va, vb),
        },
        (
            QuantizedVector::F8 { bytes: ba, min: min_a, max: max_a },
            QuantizedVector::F8 { bytes: bb, min: min_b, max: max_b },
        ) => match metric {
            DistanceMetric::DotProduct => dot_product_f8(ba, *min_a, *max_a, bb, *min_b, *max_b),
            DistanceMetric::Cosine => cosine_similarity_f8(ba, *min_a, *max_a, bb, *min_b, *max_b),
            DistanceMetric::HighPrecisionEuclidean => euclidean_distance_f8(ba, *min_a, *max_a, bb, *min_b, *max_b),
        },
        _ => Err("Asymmetric layout mismatch: Operands must share matching structural precision layouts.".to_string()),
    }
}

// ============================================================================================
// CORE F32 ATTACHED HARDWARE SIMD ROUTER PIPELINES
// ============================================================================================

/// Computes the mathematical Dot Product using architectural hardware runtime SIMD selection lanes.
///
/// Routes calculations through AVX2/FMA on x86_64 or NEON on aarch64 architectures.
fn dot_product_f32(a: &[f32], b: &[f32]) -> Result<f32, String> {
    if a.len() != b.len() {
        return Err(
            "Dimension mismatch: Vector lengths must be structurally identical.".to_string(),
        );
    }

    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        return Ok(unsafe { simd::x86::dot_product_f32(a, b) });
    }

    #[cfg(target_arch = "aarch64")]
    {
        return Ok(unsafe { simd::arm::dot_product_f32(a, b) });
    }

    #[allow(unreachable_code)]
    Ok(a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum())
}

/// Computes the Cosine Similarity by cleanly mapping magnitudes onto our unrolled dot_product engine.
fn cosine_similarity_f32(a: &[f32], b: &[f32]) -> Result<f32, String> {
    let dot = dot_product_f32(a, b)?;

    // Leverage our unrolled SIMD dot product to compute self-magnitudes (L2 norms)
    let norm_sq_a = dot_product_f32(a, a)?;
    let norm_sq_b = dot_product_f32(b, b)?;

    if norm_sq_a <= 0.0 || norm_sq_b <= 0.0 {
        return Err("Zero magnitude vector error: Vector magnitude cannot be zero during cosine normalization.".to_string());
    }

    Ok(dot / (norm_sq_a.sqrt() * norm_sq_b.sqrt()))
}

/// Computes the standard Euclidean Distance utilizing optimized multi-precision squared differential lanes.
fn euclidean_distance_f32(a: &[f32], b: &[f32]) -> Result<f32, String> {
    if a.len() != b.len() {
        return Err(
            "Dimension mismatch: Vector lengths must be structurally identical.".to_string(),
        );
    }

    #[cfg(target_arch = "x86_64")]
    // FIXED: Ensured both avx2 and fma execution flags are verified safely before dispatching to hardware modules
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        return Ok(unsafe { simd::x86::euclidean_sq_f32(a, b) }.sqrt());
    }

    #[cfg(target_arch = "aarch64")]
    {
        return Ok(unsafe { simd::arm::euclidean_sq_f32(a, b) }.sqrt());
    }

    #[allow(unreachable_code)]
    {
        let s: f32 = a
            .iter()
            .zip(b.iter())
            .map(|(&x, &y)| {
                let diff = x - y;
                diff * diff
            })
            .sum();
        Ok(s.sqrt())
    }
}

// ============================================================================================
// OPTIMIZED HALF-PRECISION F16 IMPLEMENTATIONS
// ============================================================================================

/// Computes the scalar dot product of half-precision float blocks by casting data lanes to standard f32 metrics.
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

/// Computes angular orientation properties over half-precision floats using explicit software unpack operations.
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

/// Evaluates absolute straight-line error variance across compressed half-precision spatial data arrays.
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
/// Dynamically expands byte arrays into product sums using explicit algebraic scaling transformations:
/// $Factor \cdot \sum a_i b_i + OffsetAdjustment$.
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

    let mut sums = (0u32, 0u32, 0u32); // Layout register layout: (sum_ab, sum_a, sum_b)

    // Route integer extraction loops onto vectorized AVX2 registers if detected
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") {
        unsafe {
            simd::x86::dot_product_f8(a, b, &mut sums);
        }
    } else {
        for (&x, &y) in a.iter().zip(b.iter()) {
            sums.0 += (x as u32) * (y as u32);
            sums.1 += x as u32;
            sums.2 += y as u32;
        }
    }

    #[cfg(not(target_arch = "x86_64"))]
    {
        for (&x, &y) in a.iter().zip(b.iter()) {
            sums.0 += (x as u32) * (y as u32);
            sums.1 += x as u32;
            sums.2 += y as u32;
        }
    }

    // Rehydrate integer accumulate values back into exact f32 geometric manifolds
    let factor = (range_a * range_b) / (255.0 * 255.0);
    let part1 = factor * (sums.0 as f32);
    let part2 = ((range_a * min_b) / 255.0) * (sums.1 as f32);
    let part3 = ((range_b * min_a) / 255.0) * (sums.2 as f32);
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

    // Internal lambda scaling function to compute L2 norms over packed integer buffers
    let calc_norm = |bytes: &[u8], min_val: f32, max_val: f32| -> f32 {
        let range = max_val - min_val;
        let mut sums = (0u32, 0u32, 0u32);

        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx2") {
            unsafe {
                simd::x86::dot_product_f8(bytes, bytes, &mut sums);
            }
        } else {
            for &x in bytes.iter() {
                let xi = x as u32;
                sums.0 += xi * xi;
                sums.1 += xi;
            }
        }

        #[cfg(not(target_arch = "x86_64"))]
        {
            for &x in bytes.iter() {
                let xi = x as u32;
                sums.0 += xi * xi;
                sums.1 += xi;
            }
        }

        let factor = (range * range) / (255.0 * 255.0);
        let p1 = factor * (sums.0 as f32);
        let p2 = (2.0 * range * min_val / 255.0) * (sums.1 as f32);
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
    // Sequential fallback loop unpacking compressed float intervals directly for distance sum accumulations
    for (&x, &y) in a.iter().zip(b.iter()) {
        let va = min_a + (x as f32 / 255.0) * range_a;
        let vb = min_b + (y as f32 / 255.0) * range_b;
        let diff = va - vb;
        sum_squares += diff * diff;
    }
    Ok(sum_squares.sqrt())
}
