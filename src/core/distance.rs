// src/core/distance.rs

/// Supported geometric metrics utilized for calculating spatial proximity between high-dimensional vector embeddings.
pub enum DistanceMetric {
    /// Vector dot product metric, optimal for normalized embeddings ($A \cdot B = \sum a_i b_i$).
    DotProduct,
    /// Angular cosine similarity mapping structural orientation independently of magnitude ($\frac{A \cdot B}{\|A\| \|B\|}$).
    Cosine,
    /// Absolute straight-line distance algorithm mapping geometric variance within a Cartesian space ($\sqrt{\sum (a_i - b_i)^2}$).
    HighPrecisionEuclidean,
}

/// Computes the mathematical Dot Product between two equal-length floating-point slices.
///
/// # Errors
/// Returns an error dynamic string if a structural dimension mismatch is detected between the operands.
pub fn dot_product(a: &[f32], b: &[f32]) -> Result<f32, String> {
    if a.len() != b.len() {
        return Err(
            "Dimension mismatch: Vector lengths must be structurally identical.".to_string(),
        );
    }
    Ok(a.iter().zip(b.iter()).map(|(x, y)| x * y).sum())
}

/// Computes the Cosine Similarity between two floating-point directional vectors.
///
/// # Errors
/// Returns an error if dimensions mismatch or if either vector resolves to a zero magnitude (preventing division by zero).
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> Result<f32, String> {
    if a.len() != b.len() {
        return Err(
            "Dimension mismatch: Vector lengths must be structurally identical.".to_string(),
        );
    }

    let dot = dot_product(a, b)?;
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    // Critical invariant check to prevent undefined behavior via mathematical division by zero
    if norm_a == 0.0 || norm_b == 0.0 {
        return Err("Vector magnitude mathematically cannot be zero during cosine normalization normalization steps.".to_string());
    }

    Ok(dot / (norm_a * norm_b))
}

/// Computes the standard Euclidean Distance between two analytical multi-dimensional target coordinates.
///
/// # Errors
/// Returns an error sequence if the structural layer shapes differ between the two slices.
pub fn euclidean_distance(a: &[f32], b: &[f32]) -> Result<f32, String> {
    if a.len() != b.len() {
        return Err(
            "Dimension mismatch: Vector lengths must be structurally identical.".to_string(),
        );
    }

    let sum_squares: f32 = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let diff = x - y;
            diff * diff
        })
        .sum();

    Ok(sum_squares.sqrt())
}
