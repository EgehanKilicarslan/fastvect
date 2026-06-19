// src/index/hnsw.rs

use crate::core::distance::{cosine_similarity, dot_product, euclidean_distance};
use crate::{DistanceMetric, Filter, Point};
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};

/// Represents a single distinct graphical vertex within the multi-tiered HNSW index mesh.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HNSWNode {
    /// Global tracking key linking the graphical node to its shared parent data model identity.
    pub point_id: u64,
    /// Fixed structural array layout mapping internal level layers directly to discrete neighbor collections.
    pub neighbors: [Vec<u64>; 16],
}

/// Core state machine managing properties, multi-tier routing topologies, and graph layers for the HNSW index.
#[derive(Serialize, Deserialize)]
pub struct HNSWIndex {
    /// Maximum connection limits allowed per node layer within the system matrix ($M$).
    pub m: usize,
    /// Boundary limits controlling candidate size queues tracked during graph generation ($ef_{construction}$).
    pub ef_construction: usize,
    /// Boundary limits controlling candidate evaluation queues processed during operational retrieval ($ef_{search}$).
    pub ef_search: usize,
    /// Optional top-tier global entryway node marking the spatial entry gateway into our proximity traversal pipeline.
    pub enter_node: Option<u64>,
    /// Highest index layer currently allocated inside the active graph topology bounds.
    pub max_current_level: usize,
    /// Global repository indexing unique target keys to distinct multi-layer adjacency vertex structures.
    pub nodes: FxHashMap<u64, HNSWNode>,
}

impl HNSWIndex {
    /// Instantiates an empty, pre-configured Hierarchical Navigable Small World index manager.
    pub fn new(m: usize, ef_construction: usize, ef_search: usize) -> Self {
        Self {
            m,
            ef_construction,
            ef_search,
            enter_node: None,
            max_current_level: 0,
            nodes: FxHashMap::default(),
        }
    }

    /// Evaluates a target layer using an exponential decay function to calculate a new vertex's peak height.
    fn generate_random_level(&self) -> usize {
        let r: f64 = rand::random::<f64>();
        let factor = 1.0 / (self.m as f64).ln();
        if r == 0.0 {
            return 15;
        }
        let level = (-r.ln() * factor) as usize;
        std::cmp::min(level, 15)
    }

    /// Traverses a specific layer using a greedy search approach to isolate the closest vertex node near the target query array.
    ///
    /// # Parameters
    /// * `query_vector` - High-dimensional source slice coordinate array used as the lookup query target.
    /// * `curr_obj` - Global workspace identifier matching the current entry checkpoint vertex.
    /// * `level` - The structural matrix layer index currently being traversed.
    /// * `points_ref` - Read reference link targeting the underlying shared atomic vector payload pool.
    /// * `filter` - An optional tenant boundary restriction module used to enforce isolation constraints.
    /// * `deleted_bits` - Read slice tracking historical tombstone data record deletions.
    ///
    /// # Returns
    /// The optimal structural node identifier pointing to the closest spatial vertex verified on this target plane.
    pub fn search_layer(
        &self,
        query_vector: &[f32],
        curr_obj: u64,
        level: usize,
        points_ref: &FxHashMap<u64, Point>,
        filter: Option<&Filter>,
        deleted_bits: &[bool],
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
                if level < 16 {
                    let neighbors = &node.neighbors[level];
                    for &neighbor_id in neighbors {
                        let nid_idx = neighbor_id as usize;

                        if nid_idx < deleted_bits.len() && deleted_bits[nid_idx] {
                            continue;
                        }

                        if let Some(neighbor_point) = points_ref.get(&neighbor_id) {
                            #[cfg(target_arch = "x86_64")]
                            unsafe {
                                core::arch::x86_64::_mm_prefetch(
                                    neighbor_point.vector.as_ptr() as *const i8,
                                    core::arch::x86_64::_MM_HINT_T0,
                                );
                            }

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

    /// Evaluates proximity vectors via highly optimized $O(\log N)$ structural graph traversal loops.
    ///
    /// Executes greedy macro-routing cascades descending from upper layers down to Layer 0, shifting
    /// to a local Greedy Beam Search bounded by the internal `ef_search` threshold capacity.
    ///
    /// # Parameters
    /// * `query_vector` - Target float matrix coordinates to evaluate across spatial topologies.
    /// * `limit` - The total depth matching threshold boundary (Top-K) to harvest from active memory slots.
    /// * `metric` - Configuration configurations matching supported distance parameters.
    /// * `points_ref` - Read reference link targeting the underlying shared atomic vector payload pool.
    /// * `filter` - An optional identification tag string used to enforce secure workspace isolation.
    /// * `deleted_bits` - Read slice tracking historical tombstone data record deletions.
    ///
    /// # Returns
    /// A sorted collection of final matched query outputs paired with spatial similarity scores.
    pub fn search(
        &self,
        query_vector: &[f32],
        limit: usize,
        metric: DistanceMetric,
        points_ref: &FxHashMap<u64, Point>,
        filter: Option<&Filter>,
        deleted_bits: &[bool],
    ) -> Vec<crate::storage::segment::SearchResult> {
        let enter_node = match self.enter_node {
            Some(node) => node,
            None => return Vec::new(),
        };

        let mut curr_obj = enter_node;
        for level in (1..=self.max_current_level).rev() {
            curr_obj = self.search_layer(
                query_vector,
                curr_obj,
                level,
                points_ref,
                filter,
                deleted_bits,
            );
        }

        let visited_max_id = self.nodes.keys().max().cloned().unwrap_or(0) as usize;
        let mut visited = vec![false; visited_max_id + 1];

        let mut candidates: Vec<(f32, u64)> = Vec::new();
        let mut results_pool: Vec<(f32, u64)> = Vec::new();

        if let Some(p) = points_ref.get(&curr_obj) {
            let dist = match cosine_similarity(query_vector, &p.vector) {
                Ok(sim) => 1.0 - sim,
                Err(_) => f32::MAX,
            };

            let c_idx = curr_obj as usize;
            if c_idx < deleted_bits.len() && deleted_bits[c_idx] {
                // Skip tombstone entryway
            } else {
                candidates.push((dist, curr_obj));
                results_pool.push((dist, curr_obj));
            }

            if c_idx < visited.len() {
                visited[c_idx] = true;
            }
        }

        while !candidates.is_empty() {
            candidates.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            let (_, nearest_cand_id) = candidates.remove(0);

            let furthest_result_dist = results_pool
                .iter()
                .map(|x| x.0)
                .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or(f32::MAX);

            if let Some(node) = self.nodes.get(&nearest_cand_id) {
                let neighbors = &node.neighbors[0];
                for &neighbor_id in neighbors {
                    let nid_idx = neighbor_id as usize;

                    if nid_idx < deleted_bits.len() && deleted_bits[nid_idx] {
                        continue;
                    }

                    if nid_idx < visited.len() && !visited[nid_idx] {
                        visited[nid_idx] = true;

                        if let Some(neighbor_point) = points_ref.get(&neighbor_id) {
                            #[cfg(target_arch = "x86_64")]
                            unsafe {
                                core::arch::x86_64::_mm_prefetch(
                                    neighbor_point.vector.as_ptr() as *const i8,
                                    core::arch::x86_64::_MM_HINT_T0,
                                );
                            }

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

                            if dist < furthest_result_dist || results_pool.len() < self.ef_search {
                                candidates.push((dist, neighbor_id));
                                results_pool.push((dist, neighbor_id));

                                if results_pool.len() > self.ef_search {
                                    results_pool.sort_by(|a, b| {
                                        a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)
                                    });
                                    results_pool.pop();
                                }
                            }
                        }
                    }
                }
            }
        }

        let mut final_scored_results: Vec<crate::storage::segment::SearchResult> = Vec::new();
        results_pool.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        for (dist, id) in results_pool {
            if let Some(point) = points_ref.get(&id) {
                let final_score = match metric {
                    DistanceMetric::DotProduct => {
                        dot_product(query_vector, &point.vector).unwrap_or(0.0)
                    }
                    DistanceMetric::Cosine => 1.0 - dist,
                    DistanceMetric::HighPrecisionEuclidean => {
                        euclidean_distance(query_vector, &point.vector).unwrap_or(f32::MAX)
                    }
                };

                final_scored_results.push((point.id, final_score, point.payload.clone()));
                if final_scored_results.len() == limit {
                    break;
                }
            }
        }

        match metric {
            DistanceMetric::DotProduct | DistanceMetric::Cosine => {
                final_scored_results
                    .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            }
            DistanceMetric::HighPrecisionEuclidean => {
                final_scored_results
                    .sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            }
        }
        final_scored_results
    }

    /// Safely injects a newly registered coordinate vector directly into the multi-tier spatial reference network.
    ///
    /// # Parameters
    /// * `point_id` - Unique transactional database token assigned to register the target object.
    /// * `vector` - Raw float array slice representing the underlying vector profile coordinates.
    /// * `points_ref` - Read access to global target points records.
    pub fn insert(&mut self, point_id: u64, vector: &[f32], points_ref: &FxHashMap<u64, Point>) {
        let insert_level = self.generate_random_level();
        let neighbors_array: [Vec<u64>; 16] = Default::default();

        let mut new_node = HNSWNode {
            point_id,
            neighbors: neighbors_array,
        };

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
                curr_obj = self.search_layer(vector, curr_obj, level, points_ref, None, &[]);
            }
        }

        for level in (0..=std::cmp::min(insert_level, self.max_current_level)).rev() {
            curr_obj = self.search_layer(vector, curr_obj, level, points_ref, None, &[]);

            if let Some(neighbor_node) = self.nodes.get_mut(&curr_obj) {
                let neighbors_list = &mut neighbor_node.neighbors[level];
                if neighbors_list.len() < self.m {
                    neighbors_list.push(point_id);
                }
            }
            new_node.neighbors[level].push(curr_obj);
        }

        if insert_level > self.max_current_level {
            self.max_current_level = insert_level;
            self.enter_node = Some(point_id);
        }

        self.nodes.insert(point_id, new_node);
    }
}
