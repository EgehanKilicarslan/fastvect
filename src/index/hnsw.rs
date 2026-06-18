// src/index/hnsw.rs

use crate::core::distance::{cosine_similarity, dot_product, euclidean_distance};
use crate::{DistanceMetric, Filter, Point};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Represents a distinct graphical vertex localized within the multi-tiered HNSW graph indexing mesh.
///
/// Each node serves as an analytical coordinate anchor that encapsulates adjacent routing links
/// mapped across multiple hierarchical connectivity tiers.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HNSWNode {
    /// Global tracking key linking the graphical node to its shared parent data model identity.
    pub point_id: u64,
    /// Adjacency reference tables assigning specific level indices to a discrete list of neighbor node IDs.
    pub neighbors: HashMap<usize, Vec<u64>>,
}

/// Core state machine managing properties, multi-tier routing topologies, and graph generation layers for the HNSW index.
///
/// This manager maintains the historical entry anchors, manages dynamic candidate tracking limits,
/// and oversees structural geometric scale boundaries to orchestrate high-speed proximity search pipelines.
#[derive(Serialize, Deserialize)]
pub struct HNSWIndex {
    /// Maximum structural connection limit allowed per graphical node layer within the system matrix ($M$).
    pub m: usize,
    /// Boundary limits controlling candidate size queues tracked during graph generation operations ($ef_{construction}$).
    pub ef_construction: usize,
    /// Boundary limits controlling candidate evaluation queues processed during operational retrieval steps ($ef_{search}$).
    pub ef_search: usize,
    /// Optional top-tier global entryway node marking the spatial entry gateway into our proximity traversal pipeline.
    pub enter_node: Option<u64>,
    /// Highest index layer currently allocated inside the active graph topology bounds.
    pub max_current_level: usize,
    /// Global internal repository indexing unique target keys to distinct multi-layer adjacency vertex structures.
    pub nodes: HashMap<u64, HNSWNode>,
}

impl HNSWIndex {
    /// Instantiates an empty, pre-configured Hierarchical Navigable Small World clustering manager.
    ///
    /// # Parameters
    /// * `m` - The bi-directional connection constraint factor limiting structural node degrees.
    /// * `ef_construction` - Search depth coefficient evaluated during indexing passes.
    /// * `ef_search` - Search capacity metrics tracked during active execution sweeps.
    ///
    /// # Examples
    /// ```
    /// use fastvect::HNSWIndex;
    /// let index = HNSWIndex::new(16, 64, 32);
    /// ```
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

    /// Evaluates a target layer using an exponential decay function to probabilistically calculate a new vertex's peak height.
    ///
    /// This routine utilizes an exponential decay distribution modeled via a standard skip-list normalization factor.
    /// Higher layers are exponentially less likely to be selected, ensuring a sparse log-scale administrative macro-routing layout.
    ///
    /// # Returns
    /// An integer mapping the target ceiling level boundary, capped explicitly at a maximum threshold of 16 layers.
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
    ///
    /// # Parameters
    /// * `query_vector` - High-dimensional source slice coordinate array used as the lookup query target.
    /// * `curr_obj` - Global workspace identifier matching the current entry or local checkpoint vertex.
    /// * `level` - The structural matrix layer index currently being traversed.
    /// * `points_ref` - Read reference link targeting the underlying shared atomic vector payload pool.
    /// * `filter` - An optional tenant boundary restriction module used to enforce isolation constraints.
    ///
    /// # Returns
    /// The optimal structural node identifier pointing to the closest spatial vertex verified on this target plane.
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

    /// Evaluates proximity vectors via highly optimized $O(\log N)$ structural mesh traversal loops.
    ///
    /// This routine executes greedy macro-routing cascades descending from upper layers down to Layer 0.
    /// Upon reaching the base layer, it shifts from linear fallback scans to a true localized graph
    /// traversal (Greedy Beam Search), sweeping immediate neighboring vertices up to `ef_search` limits.
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
        metric: DistanceMetric,
        points_ref: &HashMap<u64, Point>,
        filter: Option<&Filter>,
    ) -> Vec<crate::storage::segment::SearchResult> {
        // Edge case fallback: if graphical structural entryway maps are entirely blank, cleanly defer evaluation to baseline KNN routines
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

        // Phase 1: Macro-routing cascade down through upper administrative hierarchy tiers
        let mut curr_obj = enter_node;
        for level in (1..=self.max_current_level).rev() {
            curr_obj = self.search_layer(query_vector, curr_obj, level, points_ref, filter);
        }

        // Phase 2: True Layer 0 Localized Greedy Search (Beam Search bounded by ef_search)
        let mut visited = HashSet::new();
        let mut candidates: Vec<(f32, u64)> = Vec::new();
        let mut results_pool: Vec<(f32, u64)> = Vec::new();

        // Initialize state tracking using the optimal entrance hub harvested from macro-routing pipelines
        if let Some(p) = points_ref.get(&curr_obj) {
            let dist = match cosine_similarity(query_vector, &p.vector) {
                Ok(sim) => 1.0 - sim,
                Err(_) => f32::MAX,
            };
            candidates.push((dist, curr_obj));
            results_pool.push((dist, curr_obj));
            visited.insert(curr_obj);
        }

        // Greedy horizontal traversal exploration loop bounded by the ef_search saturation factors
        while !candidates.is_empty() {
            // Strategic sorting optimization to fetch the absolute closest spatial vertex element (Min-Distance)
            candidates.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            let (_, nearest_cand_id) = candidates.remove(0);

            let furthest_result_dist = results_pool
                .iter()
                .map(|x| x.0)
                .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or(f32::MAX);

            if let Some(node) = self.nodes.get(&nearest_cand_id) {
                if let Some(neighbors) = node.neighbors.get(&0) {
                    // Explicitly query base Layer 0
                    for &neighbor_id in neighbors {
                        if !visited.contains(&neighbor_id) {
                            visited.insert(neighbor_id);

                            if let Some(neighbor_point) = points_ref.get(&neighbor_id) {
                                // Multi-tenant graph routing safety gatekeeper
                                if let Some(f) = filter {
                                    if !f.matches(&neighbor_point.payload) {
                                        continue;
                                    }
                                }

                                let dist =
                                    match cosine_similarity(query_vector, &neighbor_point.vector) {
                                        Ok(sim) => 1.0 - sim,
                                        Err(_) => f32::MAX,
                                    };

                                // Expand local paths if neighbor is closer than current worst candidate or queue has open space
                                if dist < furthest_result_dist
                                    || results_pool.len() < self.ef_search
                                {
                                    candidates.push((dist, neighbor_id));
                                    results_pool.push((dist, neighbor_id));

                                    // Enforce strict upper boundary thresholds capped at ef_search parameters
                                    if results_pool.len() > self.ef_search {
                                        results_pool.sort_by(|a, b| {
                                            a.0.partial_cmp(&b.0)
                                                .unwrap_or(std::cmp::Ordering::Equal)
                                        });
                                        results_pool.pop();
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // Phase 3: Mathematical Mapping & Harvesting
        let mut final_scored_results: Vec<crate::storage::segment::SearchResult> = Vec::new();
        results_pool.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        for (dist, id) in results_pool {
            if let Some(point) = points_ref.get(&id) {
                // Defensive filter validation check enforcing tenant boundary isolation rules
                if let Some(f) = filter {
                    if !f.matches(&point.payload) {
                        continue;
                    }
                }

                // Transform inverted distances back to precise metric metrics scoring layouts
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

        // Match metric-specific ranking constraints
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
    /// The indexing process proceeds through two transactional phases:
    /// 1. **Phase 1 (Macro-routing):** High-speed cascading greedy descents down through upper layers to locate optimal entrance portals.
    /// 2. **Phase 2 (Micro-routing):** Graph stitching and bi-directional linkage assignments across valid target layers, enforcing the $M$ connection constraint limit.
    ///
    /// # Parameters
    /// * `point_id` - Unique transactional database token assigned to register the target object.
    /// * `vector` - Raw float array slice representing the underlying vector profile coordinates.
    /// * `points_ref` - System runtime pointer reference providing read access to global target points records.
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
