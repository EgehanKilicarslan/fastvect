// src/index/exact.rs

use crate::core::distance::{cosine_similarity, dot_product, euclidean_distance};
use crate::{DistanceMetric, Filter, Point, SearchResult};
use rustc_hash::FxHashMap;

/// Executes an $O(N)$ high-precision sequential evaluation scan across a targeted vector data space.
///
/// This standalone strategy iterates over a read-only checkpoint of the primary storage matrix
/// to compute exact analytical proximity metrics. It dynamically respects optional multi-tenancy
/// `Filter` constraints, dropping non-matching components before compute-heavy distance evaluations.
///
/// # Parameters
/// * `query_vector` - A floating-point slice representing the multi-dimensional search target coordinates.
/// * `limit` - The maximum number of nearest neighbor matches (top-K) to slice and return.
/// * `metric` - The spatial distance formula to execute during geometric evaluation sweeps.
/// * `points` - A reference link pointing to the unmutated storage mapping containing registered database entities.
/// * `filter` - An optional multi-tenancy restriction boundary used to isolate vector partitions.
/// * `deleted_bits` - A slice tracking tombstone data record deletions to bypass soft-deleted points.
///
/// # Returns
/// A sorted vector collection containing matched indices paired with their mathematical proximity scores and structural metadata.
pub fn search_exact_knn(
    query_vector: &[f32],
    limit: usize,
    metric: DistanceMetric,
    points: &FxHashMap<u64, Point>,
    filter: Option<&Filter>,
    deleted_bits: &[bool],
) -> Vec<SearchResult> {
    let mut scored_results: Vec<SearchResult> = Vec::new();

    for point in points.values() {
        let idx = point.id as usize;

        // Skip immediately if the point has been marked as deleted via tombstone bitset
        if idx < deleted_bits.len() && deleted_bits[idx] {
            continue;
        }

        // SINGLE-STAGE PRE-FILTERING: Drop non-matching tenants immediately to bypass distance overheads
        if let Some(f) = filter {
            if !f.matches(&point.payload) {
                continue;
            }
        }

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
        DistanceMetric::DotProduct | DistanceMetric::Cosine => {
            scored_results
                .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        }
        DistanceMetric::HighPrecisionEuclidean => {
            scored_results
                .sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        }
    }

    scored_results.into_iter().take(limit).collect()
}
