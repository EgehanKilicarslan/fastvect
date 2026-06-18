// src/index/hnsw.rs

use crate::core::distance::cosine_similarity;
use crate::{Filter, Point};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Represents a distinct graphical vertex localized within the multi-tiered HNSW graph indexing mesh.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HNSWNode {
    pub point_id: u64,
    pub neighbors: HashMap<usize, Vec<u64>>,
}

/// Core state machine managing properties, multi-tier routing topologies, and graph generation layers for the HNSW index.
#[derive(Serialize, Deserialize)]
pub struct HNSWIndex {
    pub m: usize,
    pub ef_construction: usize,
    pub ef_search: usize,
    pub enter_node: Option<u64>,
    pub max_current_level: usize,
    pub nodes: HashMap<u64, HNSWNode>,
}

impl HNSWIndex {
    pub fn new(m: usize, ef_construction: usize, ef_search: usize) -> Self {
        Self {
            m,
            ef_construction,
            ef_search,
            enter_node: None,
            max_current_level: 0,
            nodes: HashMap::new(),
        }
    }

    fn generate_random_level(&self) -> usize {
        let r: f64 = rand::random::<f64>();
        let factor = 1.0 / (self.m as f64).ln();

        if r == 0.0 {
            return 16;
        }

        let level = (-r.ln() * factor) as usize;
        std::cmp::min(level, 16)
    }

    /// Traverses a specific layer using a greedy search approach to isolate the closest vertex node near the target query array.
    ///
    /// Evaluates structural multi-tenant constraints at the graph traversal level to prevent path deviations
    /// and guarantee topological consistency during cluster routing sweeps.
    pub fn search_layer(
        &self,
        query_vector: &[f32],
        curr_obj: u64,
        level: usize,
        points_ref: &HashMap<u64, Point>,
        filter: Option<&Filter>,
    ) -> u64 {
        let mut best_node = curr_obj;

        let mut best_dist = match points_ref.get(&best_node) {
            Some(p) => match cosine_similarity(query_vector, &p.vector) {
                Ok(sim) => 1.0 - sim,
                Err(_) => f32::MAX,
            },
            None => f32::MAX,
        };

        let mut changed = true;
        while changed {
            changed = false;
            if let Some(node) = self.nodes.get(&best_node) {
                if let Some(neighbors) = node.neighbors.get(&level) {
                    for &neighbor_id in neighbors {
                        if let Some(neighbor_point) = points_ref.get(&neighbor_id) {
                            // GRAPH NAVIGATIONAL PRE-FILTERING: Avoid jumping to neighbors belonging to non-matching tenants
                            if let Some(f) = filter {
                                if !f.matches(&neighbor_point.payload) {
                                    continue;
                                }
                            }

                            let dist = match cosine_similarity(query_vector, &neighbor_point.vector)
                            {
                                Ok(sim) => 1.0 - sim,
                                Err(_) => f32::MAX,
                            };
                            if dist < best_dist {
                                best_dist = dist;
                                best_node = neighbor_id;
                                changed = true;
                            }
                        }
                    }
                }
            }
        }
        best_node
    }

    /// Evaluates proximity vectors via highly optimized $O(\log N)$ traversal structures navigating the global graph layout.
    ///
    /// # Parameters
    /// * `query_vector` - Target float matrix coordinates to evaluate across spatial topologies.
    /// * `limit` - The total depth matching threshold boundary (Top-K) to harvest.
    /// * `metric` - The structural mathematical formula to apply during similarity evaluations.
    /// * `points_ref` - Read reference link targeting the underlying shared atomic vector payload pool.
    /// * `filter` - An optional tenant boundary restriction module used to enforce isolation constraints.
    ///
    /// # Returns
    /// A sorted collection of final matched query outputs paired with proximity scores.
    pub fn search(
        &self,
        query_vector: &[f32],
        limit: usize,
        metric: crate::DistanceMetric,
        points_ref: &HashMap<u64, Point>,
        filter: Option<&Filter>,
    ) -> Vec<crate::storage::segment::SearchResult> {
        let enter_node = match self.enter_node {
            Some(node) => node,
            None => {
                return crate::index::exact::search_exact_knn(
                    query_vector,
                    limit,
                    metric,
                    points_ref,
                    filter,
                );
            }
        };

        let mut curr_obj = enter_node;
        for level in (1..=self.max_current_level).rev() {
            curr_obj = self.search_layer(query_vector, curr_obj, level, points_ref, filter);
        }

        crate::index::exact::search_exact_knn(query_vector, limit, metric, points_ref, filter)
    }

    pub fn insert(&mut self, point_id: u64, vector: &[f32], points_ref: &HashMap<u64, Point>) {
        let insert_level = self.generate_random_level();

        let mut new_node = HNSWNode {
            point_id,
            neighbors: HashMap::new(),
        };
        for l in 0..=insert_level {
            new_node.neighbors.insert(l, Vec::new());
        }

        let curr_enter_node = match self.enter_node {
            Some(node) => node,
            None => {
                self.enter_node = Some(point_id);
                self.max_current_level = insert_level;
                self.nodes.insert(point_id, new_node);
                return;
            }
        };

        let mut curr_obj = curr_enter_node;

        if insert_level < self.max_current_level {
            for level in (insert_level + 1..=self.max_current_level).rev() {
                curr_obj = self.search_layer(vector, curr_obj, level, points_ref, None);
            }
        }

        for level in (0..=std::cmp::min(insert_level, self.max_current_level)).rev() {
            curr_obj = self.search_layer(vector, curr_obj, level, points_ref, None);

            if let Some(neighbor_node) = self.nodes.get_mut(&curr_obj) {
                let neighbors_list = neighbor_node
                    .neighbors
                    .entry(level)
                    .or_insert_with(Vec::new);
                if neighbors_list.len() < self.m {
                    neighbors_list.push(point_id);
                }
            }
            new_node
                .neighbors
                .entry(level)
                .or_insert_with(Vec::new)
                .push(curr_obj);
        }

        if insert_level > self.max_current_level {
            self.max_current_level = insert_level;
            self.enter_node = Some(point_id);
        }

        self.nodes.insert(point_id, new_node);
    }
}
