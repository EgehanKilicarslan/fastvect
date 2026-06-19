// src/index/hnsw.rs

use crate::core::distance::compute_distance;
use crate::{DistanceMetric, Filter, Point, QuantizedVector};
use rustc_hash::{FxHashMap, FxHashSet};
use serde::{Deserialize, Serialize};
use std::cmp::Reverse;
use std::collections::BinaryHeap;

/// Floating-point wrapper providing total ordering capabilities to float metrics inside collections.
///
/// Bypasses the lack of standard `Ord` implementation for raw `f32` types in Rust, enabling
/// predictable sorting and safe operations inside state-dependent heap structures.
#[derive(PartialEq, PartialOrd)]
struct OrdF32(f32);
impl Eq for OrdF32 {}
impl Ord for OrdF32 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0
            .partial_cmp(&other.0)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

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
    /// * `metric` - Target spatial calculation distance formula configuration.
    /// * `points_ref` - Read reference pointer targeting the unmutated primary database record repository.
    /// * `filter` - Optional tenant boundary restriction configuration module.
    /// * `deleted_bits` - Dense validation slice tracking historical tombstone bit indicators.
    ///
    /// # Returns
    /// The optimal point identifier matching the closest spatial neighbor discovered on this level track.
    pub fn search_layer(
        &self,
        query_vector: &QuantizedVector,
        curr_obj: u64,
        level: usize,
        metric: DistanceMetric,
        points_ref: &FxHashMap<u64, Point>,
        filter: Option<&Filter>,
        deleted_bits: &[bool],
    ) -> u64 {
        let mut best_node = curr_obj;
        let mut best_dist = match points_ref.get(&best_node) {
            // FIXED: Propagated the true caller target metric consistently into layer traversals
            Some(p) => match compute_distance(query_vector, &p.vector, metric) {
                Ok(sim) => {
                    if metric == DistanceMetric::Cosine {
                        1.0 - sim
                    } else {
                        sim
                    }
                }
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
                            // FIXED: Extracted direct inner array heap data memory address for prefetch instructions
                            #[cfg(target_arch = "x86_64")]
                            unsafe {
                                let data_ptr = match &neighbor_point.vector {
                                    QuantizedVector::F32(v) => v.as_ptr() as *const i8,
                                    QuantizedVector::F16(v) => v.as_ptr() as *const i8,
                                    QuantizedVector::F8 { bytes, .. } => {
                                        bytes.as_ptr() as *const i8
                                    }
                                };
                                core::arch::x86_64::_mm_prefetch(
                                    data_ptr,
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
                                metric,
                            ) {
                                Ok(sim) => {
                                    if metric == DistanceMetric::Cosine {
                                        1.0 - sim
                                    } else {
                                        sim
                                    }
                                }
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
    pub fn search(
        &self,
        query_vector: &QuantizedVector,
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

        // Macro-Traversing Step: Descend aggressively across upper layers to identify the optimal entryway hub.
        let mut curr_obj = enter_node;

        // FIXED: Upper routing layers are completely filter-free to preserve optimal graph recall navigation!
        for level in (1..=self.max_current_level).rev() {
            curr_obj = self.search_layer(
                query_vector,
                curr_obj,
                level,
                metric,
                points_ref,
                None,
                deleted_bits,
            );
        }

        // FIXED: Transitioned to a localized dense FxHashSet tracking layout to completely solve memory OOM DoS risk
        let mut visited =
            FxHashSet::with_capacity_and_hasher(self.ef_search * 4, Default::default());

        // FIXED: Replaced legacy sorted Vec structure with highly efficient binary Min/Max-Heaps
        let mut candidates: BinaryHeap<Reverse<(OrdF32, u64)>> = BinaryHeap::new();
        let mut results: BinaryHeap<(OrdF32, u64)> = BinaryHeap::new();

        if let Some(p) = points_ref.get(&curr_obj) {
            let dist = match compute_distance(query_vector, &p.vector, metric) {
                Ok(sim) => {
                    if metric == DistanceMetric::Cosine {
                        1.0 - sim
                    } else {
                        sim
                    }
                }
                Err(_) => f32::MAX,
            };

            let c_idx = curr_obj as usize;
            if c_idx < deleted_bits.len() && deleted_bits[c_idx] {
                // Skip processing soft-deleted entryway checkpoints.
            } else {
                visited.insert(curr_obj);
                candidates.push(Reverse((OrdF32(dist), curr_obj)));

                // FIXED: Entryway validation boundary mended securely to block logical workspace leaks
                let passes_filter = filter.map_or(true, |f| f.matches(&p.payload));
                if passes_filter {
                    results.push((OrdF32(dist), curr_obj));
                }
            }
        }

        // Micro-Traversing Step: Execute dynamic local greedy beam search across layer 0 boundaries.
        while let Some(Reverse((cand_dist, nearest_cand_id))) = candidates.pop() {
            // FIXED: Integrated classic HNSW dynamic convergence termination to secure logarithmic lookup complexities
            let worst_result_dist = results.peek().map(|(d, _)| d.0).unwrap_or(f32::MAX);
            if cand_dist.0 > worst_result_dist && results.len() >= self.ef_search {
                break;
            }

            if let Some(node) = self.nodes.get(&nearest_cand_id) {
                let neighbors = &node.neighbors[0];
                for &neighbor_id in neighbors {
                    let nid_idx = neighbor_id as usize;

                    if nid_idx < deleted_bits.len() && deleted_bits[nid_idx] {
                        continue;
                    }

                    if visited.insert(neighbor_id) {
                        if let Some(neighbor_point) = points_ref.get(&neighbor_id) {
                            #[cfg(target_arch = "x86_64")]
                            unsafe {
                                let data_ptr = match &neighbor_point.vector {
                                    QuantizedVector::F32(v) => v.as_ptr() as *const i8,
                                    QuantizedVector::F16(v) => v.as_ptr() as *const i8,
                                    QuantizedVector::F8 { bytes, .. } => {
                                        bytes.as_ptr() as *const i8
                                    }
                                };
                                core::arch::x86_64::_mm_prefetch(
                                    data_ptr,
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
                                metric,
                            ) {
                                Ok(sim) => {
                                    if metric == DistanceMetric::Cosine {
                                        1.0 - sim
                                    } else {
                                        sim
                                    }
                                }
                                Err(_) => f32::MAX,
                            };

                            let current_worst =
                                results.peek().map(|(d, _)| d.0).unwrap_or(f32::MAX);
                            if dist < current_worst || results.len() < self.ef_search {
                                candidates.push(Reverse((OrdF32(dist), neighbor_id)));
                                results.push((OrdF32(dist), neighbor_id));

                                if results.len() > self.ef_search {
                                    results.pop(); // Evicts the furthest element inside O(log ef) steps
                                }
                            }
                        }
                    }
                }
            }
        }

        // Post-Processing Step: Materialize, calibrate, and transform results into target metric distances.
        let mut final_scored_results: Vec<crate::storage::segment::SearchResult> =
            Vec::with_capacity(limit.min(results.len()));
        let mut sorted_pool = results.into_sorted_vec();
        sorted_pool.reverse(); // Bring the closest matching elements to the absolute front line indices

        for (OrdF32(dist), id) in sorted_pool.into_iter().take(limit) {
            if let Some(point) = points_ref.get(&id) {
                let final_score = if metric == DistanceMetric::Cosine {
                    1.0 - dist
                } else {
                    dist
                };
                final_scored_results.push((point.id, final_score, point.payload.clone()));
            }
        }
        final_scored_results
    }

    /// Safely injects a multi-precision quantized vector model directly into the multi-tier graph mesh network.
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
        let empty_deleted_bits = Vec::new();

        // Routing Cascade Pass: Traverse down from max peak towards insertion level heights.
        if insert_level < self.max_current_level {
            for level in (insert_level + 1..=self.max_current_level).rev() {
                curr_obj = self.search_layer(
                    vector,
                    curr_obj,
                    level,
                    DistanceMetric::Cosine, // Native graph architecture metric configuration
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
                DistanceMetric::Cosine,
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
