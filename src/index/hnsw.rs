// src/index/hnsw.rs

use crate::core::distance::compute_distance;
use crate::{DistanceMetric, Filter, Point, QuantizedVector};
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
    ///
    /// # Parameters
    /// * `m` - The maximum connection degree bound allocated per single graph node.
    /// * `ef_construction` - Search queue capacity threshold utilized during index construction steps.
    /// * `ef_search` - Dynamic evaluation beam search width constraint processed during retrieval passes.
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

    /// Evaluates a dynamic level assignment using a logarithmic exponential decay distribution function.
    ///
    /// Normalizes node allocation properties to build scaling hierarchies that govern logarithmic graph navigation.
    fn generate_random_level(&self) -> usize {
        let r: f64 = rand::random::<f64>();
        let factor = 1.0 / (self.m as f64).ln();
        if r == 0.0 {
            return 15;
        }
        let level = (-r.ln() * factor) as usize;
        std::cmp::min(level, 15)
    }

    /// Traverses a specific hierarchical layer plane using a greedy heuristic approach to isolate the closest local vertex.
    ///
    /// # Parameters
    /// * `query_vector` - Dynamic target multi-precision quantized vector to evaluate.
    /// * `curr_obj` - Active checkpoint identifier representing the entry target node for this layer plane.
    /// * `level` - The discrete structural matrix layer height index to query.
    /// * `points_ref` - Read reference pointer targeting the unmutated primary database record repository.
    /// * `filter` - Optional tenant boundary restriction configuration module.
    /// * `deleted_bits` - Dense validation vector tracking historical tombstone bit indicators under lock-free parameters.
    ///
    /// # Returns
    /// The optimal point identifier matching the closest spatial neighbor discovered on this level track.
    pub fn search_layer(
        &self,
        query_vector: &QuantizedVector,
        curr_obj: u64,
        level: usize,
        points_ref: &FxHashMap<u64, Point>,
        filter: Option<&Filter>,
        deleted_bits: &Vec<bool>,
    ) -> u64 {
        let mut best_node = curr_obj;
        let mut best_dist = match points_ref.get(&best_node) {
            Some(p) => match compute_distance(query_vector, &p.vector, DistanceMetric::Cosine) {
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

                        // Check tombstones instantly to prevent traversing soft-deleted segments
                        if nid_idx < deleted_bits.len() && deleted_bits[nid_idx] {
                            continue;
                        }

                        if let Some(neighbor_point) = points_ref.get(&neighbor_id) {
                            // Issue explicit x86 execution hardware cache hints to fetch adjacent data lines early
                            #[cfg(target_arch = "x86_64")]
                            unsafe {
                                core::arch::x86_64::_mm_prefetch(
                                    &neighbor_point.vector as *const _ as *const i8,
                                    core::arch::x86_64::_MM_HINT_T0,
                                );
                            }

                            // Enforce metadata single-stage filtering prior to expensive mathematical distance runs
                            if let Some(f) = filter {
                                if !f.matches(&neighbor_point.payload) {
                                    continue;
                                }
                            }

                            let dist = match compute_distance(
                                query_vector,
                                &neighbor_point.vector,
                                DistanceMetric::Cosine,
                            ) {
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

    /// Evaluates dynamic proximity targets across graph hierarchies via optimized $O(\log N)$ traversal paths.
    ///
    /// Executes greedy macro-routing cascades descending from allocated top layers down to layer 0, shifting
    /// smoothly to a localized dynamic beam search bounded strictly by the configured `ef_search` limits.
    ///
    /// # Parameters
    /// * `query_vector` - Dynamic target multi-precision quantized vector to evaluate across spatial graphs.
    /// * `limit` - Total top-K matched entries depth window to collect and slice.
    /// * `metric` - Targeted distance proximity metric formula configuration.
    /// * `points_ref` - Shared read pointer linking the query framework to active memory partition maps.
    /// * `filter` - Optional tenant isolation restriction constraint module.
    /// * `deleted_bits` - Dense validation vector tracking historical tombstone bit indicators under lock-free parameters.
    ///
    /// # Returns
    /// A sorted vector collection containing matched proximity results paired with operational distance metrics.
    pub fn search(
        &self,
        query_vector: &QuantizedVector,
        limit: usize,
        metric: DistanceMetric,
        points_ref: &FxHashMap<u64, Point>,
        filter: Option<&Filter>,
        deleted_bits: &Vec<bool>,
    ) -> Vec<crate::storage::segment::SearchResult> {
        let enter_node = match self.enter_node {
            Some(node) => node,
            None => return Vec::new(),
        };

        // Macro-Traversing Step: Descend aggressively across upper layers to identify the optimal entryway hub.
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
            let dist = match compute_distance(query_vector, &p.vector, DistanceMetric::Cosine) {
                Ok(sim) => 1.0 - sim,
                Err(_) => f32::MAX,
            };

            let c_idx = curr_obj as usize;
            if c_idx < deleted_bits.len() && deleted_bits[c_idx] {
                // Skip soft-deleted entryway checkpoints.
            } else {
                // Entryway is a valid routing candidate to jump further across network nodes
                candidates.push((dist, curr_obj));

                // FIXED: Enforce strict single-stage metadata verification before letting the
                // entryway node inject itself into the final results pool tracker to prevent topological leakage!
                let mut is_match = true;
                if let Some(f) = filter {
                    if !f.matches(&p.payload) {
                        is_match = false;
                    }
                }

                if is_match {
                    results_pool.push((dist, curr_obj));
                }
            }

            if c_idx < visited.len() {
                visited[c_idx] = true;
            }
        }

        // Micro-Traversing Step: Execute dynamic local greedy beam search across layer 0 boundaries.
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
                                    &neighbor_point.vector as *const _ as *const i8,
                                    core::arch::x86_64::_MM_HINT_T0,
                                );
                            }

                            if let Some(f) = filter {
                                if !f.matches(&neighbor_point.payload) {
                                    continue;
                                }
                            }

                            let dist = match compute_distance(
                                query_vector,
                                &neighbor_point.vector,
                                DistanceMetric::Cosine,
                            ) {
                                Ok(sim) => 1.0 - sim,
                                Err(_) => f32::MAX,
                            };

                            if dist < furthest_result_dist || results_pool.len() < self.ef_search {
                                candidates.push((dist, neighbor_id));
                                results_pool.push((dist, neighbor_id));

                                // Bound the result pool matching target search widths
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

        // Post-Processing Step: Materialize, calibrate, and transform results into target metric distances.
        let mut final_scored_results: Vec<crate::storage::segment::SearchResult> = Vec::new();
        results_pool.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        for (dist, id) in results_pool {
            if let Some(point) = points_ref.get(&id) {
                let final_score = match metric {
                    DistanceMetric::DotProduct => {
                        compute_distance(query_vector, &point.vector, DistanceMetric::DotProduct)
                            .unwrap_or(0.0)
                    }
                    DistanceMetric::Cosine => 1.0 - dist,
                    DistanceMetric::HighPrecisionEuclidean => compute_distance(
                        query_vector,
                        &point.vector,
                        DistanceMetric::HighPrecisionEuclidean,
                    )
                    .unwrap_or(f32::MAX),
                };

                final_scored_results.push((point.id, final_score, point.payload.clone()));
                if final_scored_results.len() == limit {
                    break;
                }
            }
        }

        // Apply dynamic sorting passes tailored to the mathematical metric constraints
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

    /// Safely injects a multi-precision quantized vector model directly into the multi-tier graph mesh network.
    ///
    /// Manages edge linking routines across randomized insertion heights, maintaining dynamic clustering
    /// connectivity thresholds under concurrent workload patterns.
    ///
    /// # Parameters
    /// * `point_id` - Unique key identifier targeting the entity registry mapping.
    /// * `vector` - Reference link to the newly quantized data entity coordinates package.
    /// * `points_ref` - Read access pointing directly onto the master records storage pool.
    pub fn insert(
        &mut self,
        point_id: u64,
        vector: &QuantizedVector,
        points_ref: &FxHashMap<u64, Point>,
    ) {
        let insert_level = self.generate_random_level();
        let neighbors_array: [Vec<u64>; 16] = Default::default();

        let mut new_node = HNSWNode {
            point_id,
            neighbors: neighbors_array,
        };

        // Cold-Start Invariant: Initialize master entry nodes if the graph network is empty.
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

        // Created a localized dummy empty Vec to satisfy the lock-free type signatures without allocations
        let empty_deleted_bits = Vec::new();

        // Routing Cascade Pass: Traverse down from max peak towards insertion level heights.
        if insert_level < self.max_current_level {
            for level in (insert_level + 1..=self.max_current_level).rev() {
                curr_obj = self.search_layer(
                    vector,
                    curr_obj,
                    level,
                    points_ref,
                    None,
                    &empty_deleted_bits,
                );
            }
        }

        // Connection Stitching Pass: Connect and weave bilateral adjacency lists matching degree limit matrix configurations.
        for level in (0..=std::cmp::min(insert_level, self.max_current_level)).rev() {
            curr_obj = self.search_layer(
                vector,
                curr_obj,
                level,
                points_ref,
                None,
                &empty_deleted_bits,
            );

            if let Some(neighbor_node) = self.nodes.get_mut(&curr_obj) {
                let neighbors_list = &mut neighbor_node.neighbors[level];
                if neighbors_list.len() < self.m {
                    neighbors_list.push(point_id);
                }
            }
            new_node.neighbors[level].push(curr_obj);
        }

        // Adjust entry landmarks if the generated node transcends active height metrics.
        if insert_level > self.max_current_level {
            self.max_current_level = insert_level;
            self.enter_node = Some(point_id);
        }

        self.nodes.insert(point_id, new_node);
    }
}
