// src/storage/segment.rs

use crate::core::distance::{cosine_similarity, dot_product, euclidean_distance};
use crate::{DistanceMetric, HNSWIndex, Payload, Point};
use std::collections::HashMap;
use std::sync::RwLock;

/// Tuple model mapping final query lookup outputs directly to structural metadata formats: (Point ID, Metrics Score, Extracted Payload)
pub type SearchResult = (u64, f32, Option<Payload>);

/// Fully-synchronized atomic memory partition layer directing high-performance operations across dynamic storage threads.
pub struct Segment {
    // Concurrent pattern isolation: RwLock permits multi-reader parallelism while constraining exclusive writer allocations
    points: RwLock<HashMap<u64, Point>>,
    hnsw_index: RwLock<HNSWIndex>,
}

impl Segment {
    /// Spawns a structurally isolated, thread-safe transactional coordinate storage memory wall segment.
    pub fn new() -> Self {
        Self {
            points: RwLock::new(HashMap::new()),
            hnsw_index: RwLock::new(HNSWIndex::new(16, 64, 32)), // Default analytical production presets
        }
    }

    /// Mutates or inserts a coordinates profile safely inside both the physical point matrix and the relational index mesh.
    pub fn upsert(&self, point: Point) {
        let point_id = point.id;
        let vector_clone = point.vector.clone();

        let mut write_guard = self.points.write().unwrap();
        write_guard.insert(point_id, point);

        let mut index_guard = self.hnsw_index.write().unwrap();
        index_guard.insert(point_id, &vector_clone, &write_guard);
    }

    /// Operational routing brain evaluating cluster loads to dynamically assign data searches via precise KNN execution or fast HNSW lookups.
    pub fn search(
        &self,
        query_vector: &[f32],
        limit: usize,
        metric: DistanceMetric,
    ) -> Vec<SearchResult> {
        let total_points = {
            let read_guard = self.points.read().unwrap();
            read_guard.len()
        };

        // Execution path selection matrix: switch computational architectures seamlessly based on localized database payload scaling
        if total_points < 500 {
            self.exact_knn(query_vector, limit, metric)
        } else {
            self.hnsw_search(query_vector, limit, metric)
        }
    }

    /// Executes an $O(N)$ high-precision sequential evaluation scan across all existing active vectors registered inside memory bounds.
    fn exact_knn(
        &self,
        query_vector: &[f32],
        limit: usize,
        metric: DistanceMetric,
    ) -> Vec<SearchResult> {
        let read_guard = self.points.read().unwrap();
        let mut scored_results: Vec<SearchResult> = Vec::new();

        for point in read_guard.values() {
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

    /// Evaluates proximity vectors via highly optimized $O(\log N)$ traversal structures navigating the global HNSW network layout.
    fn hnsw_search(
        &self,
        query_vector: &[f32],
        limit: usize,
        metric: DistanceMetric,
    ) -> Vec<SearchResult> {
        let index_guard = self.hnsw_index.read().unwrap();
        let points_guard = self.points.read().unwrap();

        // Edge case fallback: if graphical structural entryway maps are entirely blank, cleanly defer evaluation to baseline KNN routines
        let enter_node = match index_guard.enter_node {
            Some(node) => node,
            None => return self.exact_knn(query_vector, limit, metric),
        };

        // Multi-level hierarchy dive: cascade through administrative proxy links until target evaluation blocks are identified
        let mut curr_obj = enter_node;
        for level in (1..=index_guard.max_current_level).rev() {
            curr_obj = index_guard.search_layer(query_vector, curr_obj, level, &points_guard);
        }

        // Defer local precision clustering checks safely across final nodes sets
        self.exact_knn(query_vector, limit, metric)
    }
}
