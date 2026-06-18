// src/index/exact.rs

use crate::core::distance::{cosine_similarity, dot_product, euclidean_distance};
use crate::{DistanceMetric, Point, SearchResult};
use std::collections::HashMap;

/// Executes an $O(N)$ high-precision sequential evaluation scan across a targeted vector data space.
///
/// This standalone strategy iterates over a read-only checkpoint of the primary storage matrix
/// to compute exact analytical proximity metrics. It serves as the definitive ground-truth benchmark
/// engine used to calculate recall parameters and acts as the low-cardinality fallback lane.
///
/// # Parameters
/// * `query_vector` - A floating-point slice representing the multi-dimensional search target coordinates.
/// * `limit` - The maximum number of nearest neighbor matches (top-K) to slice and return.
/// * `metric` - The spatial distance formula to execute during geometric evaluation sweeps.
/// * `points` - A reference link pointing to the unmutated storage mapping containing registered database entities.
///
/// # Returns
/// A sorted vector collection containing matched indices paired with their mathematical proximity scores and structural metadata.
pub fn search_exact_knn(
    query_vector: &[f32],
    limit: usize,
    metric: DistanceMetric,
    points: &HashMap<u64, Point>,
) -> Vec<SearchResult> {
    let mut scored_results: Vec<SearchResult> = Vec::new();

    for point in points.values() {
        let score_res = match metric {
            DistanceMetric::DotProduct => dot_product(query_vector, &point.vector),
            DistanceMetric::Cosine => cosine_similarity(query_vector, &point.vector),
            DistanceMetric::HighPrecisionEuclidean => {
                euclidean_distance(query_vector, &point.vector)
            }
        };

        if let Ok(score) = score_res {
            scored_results.push((point.id, score, point.payload.clone()));
        }
    }

    // Strategic sorting pass resolving mathematical properties native to individual metrics parameters
    match metric {
        // Similarity maximization: higher values dictate proximal optimization parameters
        DistanceMetric::DotProduct | DistanceMetric::Cosine => {
            scored_results
                .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        }
        // Error variance minimization: lower mathematical bounds prove spatial optimization parameters
        DistanceMetric::HighPrecisionEuclidean => {
            scored_results
                .sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        }
    }

    scored_results.into_iter().take(limit).collect()
}
