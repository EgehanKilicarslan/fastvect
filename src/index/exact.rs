// src/index/exact.rs

use crate::core::distance::compute_distance;
use crate::{DistanceMetric, Filter, Point, QuantizedVector, SearchResult};
use rustc_hash::FxHashMap;

/// Executes an $O(N)$ high-precision sequential evaluation scan across a targeted vector data space.
///
/// This synchronous scanning pipeline iterates directly over a read-only checkpoint of the primary
/// storage matrix to compute exact geometric proximity. It honors real-time multi-tenancy
/// `Filter` constraints, dropping non-matching components early to bypass expensive metric evaluations.
///
/// # Parameters
/// * `query_vector` - Reference to the multi-precision quantized search target embedding array.
/// * `limit` - The maximum number of nearest neighbor matches (top-K) to slice and return.
/// * `metric` - The spatial proximity distance formula configuration to execute during evaluation sweeps.
/// * `points` - Reference to the unmutated primary storage map containing registered database entities.
/// * `filter` - An optional tenant isolation constraint module used to enforce security boundaries.
/// * `deleted_bits` - A dense boolean slice tracking operational tombstone bit markers to skip soft-deleted records.
///
/// # Returns
/// A sorted collection vector containing matched point identifiers paired with their explicit
/// proximity scores and rehydrated structural payload metadata.
pub fn search_exact_knn(
    query_vector: &QuantizedVector,
    limit: usize,
    metric: DistanceMetric,
    points: &FxHashMap<u64, Point>,
    filter: Option<&Filter>,
    deleted_bits: &[bool],
) -> Vec<SearchResult> {
    let mut scored_results: Vec<SearchResult> = Vec::new();

    for point in points.values() {
        let idx = point.id as usize;

        // Skip immediately if the target node index has been flagged by a soft tombstone transaction.
        if idx < deleted_bits.len() && deleted_bits[idx] {
            continue;
        }

        // SINGLE-STAGE PRE-FILTERING: Drop non-matching tenancy attributes immediately
        // to prevent wasteful down-stream distance calculation overheads.
        if let Some(f) = filter {
            if !f.matches(&point.payload) {
                continue;
            }
        }

        // Delegate proximity computation to the polymorphic accelerated distance dispatch engine.
        if let Ok(score) = compute_distance(query_vector, &point.vector, metric) {
            scored_results.push((point.id, score, point.payload.clone()));
        }
    }

    // Strategic sorting pass resolving mathematical properties native to individual metric parameters.
    // Similarity-based workflows sort descending, while Cartesian errors sort ascending.
    match metric {
        DistanceMetric::DotProduct | DistanceMetric::Cosine => {
            scored_results
                .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        }
        DistanceMetric::HighPrecisionEuclidean => {
            scored_results
                .sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        }
    }

    // Extract the requested nearest neighbors window matching the target depth threshold boundaries.
    scored_results.into_iter().take(limit).collect()
}
