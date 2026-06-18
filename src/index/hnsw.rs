// src/index/hnsw.rs

use crate::Point;
use crate::core::distance::cosine_similarity;
use std::collections::HashMap;

/// Represents a distinct graphical vertex localized within the multi-tiered HNSW graph indexing mesh.
#[derive(Clone, Debug)]
pub struct HNSWNode {
    /// Global tracking key linking the graphical node to its shared parent data model identity.
    pub point_id: u64,
    /// Adjacency reference tables assigning specific level indices to discrete list of neighbor node IDs.
    pub neighbors: HashMap<usize, Vec<u64>>,
}

/// Core state machine managing properties, multi-tier routing topologies, and graph generation layers for the HNSW index.
pub struct HNSWIndex {
    /// Maximum structural connection limit allowed per graphical node layer within the system matrix.
    pub m: usize,
    /// Boundary limits controlling candidate size queues tracked during graph generation operations.
    pub ef_construction: usize,
    /// Boundary limits controlling candidate evaluation queues processed during operational retrieval steps.
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
    fn generate_random_level(&self) -> usize {
        // Safe runtime float lookup independent of custom environment imports
        let r: f64 = rand::random::<f64>();
        let factor = 1.0 / (self.m as f64).ln();

        // Safety fallback guard covering boundary limits
        if r == 0.0 {
            return 16;
        }

        let level = (-r.ln() * factor) as usize;
        std::cmp::min(level, 16) // Explicitly capped at standard optimization heights
    }

    /// Traverses a specific layer using a greedy search approach to isolate the closest vertex node near the target query array.
    pub fn search_layer(
        &self,
        query_vector: &[f32],
        curr_obj: u64,
        level: usize,
        points_ref: &HashMap<u64, Point>,
    ) -> u64 {
        let mut best_node = curr_obj;

        // HNSW naturally works via minimization. We convert cosine similarity to angular distance bounds via (1.0 - sim)
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
                            let dist = match cosine_similarity(query_vector, &neighbor_point.vector)
                            {
                                Ok(sim) => 1.0 - sim,
                                Err(_) => f32::MAX,
                            };
                            // Convergence lock: track if a closer geometric vector has been found
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

    /// Safely injects a newly registered coordinate vector directly into the multi-tier spatial reference network.
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
                // Cold-start fallback logic: establish structural index anchors if tracking map is entirely blank
                self.enter_node = Some(point_id);
                self.max_current_level = insert_level;
                self.nodes.insert(point_id, new_node);
                return;
            }
        };

        let mut curr_obj = curr_enter_node;

        // Phase 1: High-speed greedy macro-routing across upper administrative hierarchy tiers
        if insert_level < self.max_current_level {
            for level in (insert_level + 1..=self.max_current_level).rev() {
                curr_obj = self.search_layer(vector, curr_obj, level, points_ref);
            }
        }

        // Phase 2: Micro-routing graph stitching and bidirectional linkage assignments across active target ranges
        for level in (0..=std::cmp::min(insert_level, self.max_current_level)).rev() {
            curr_obj = self.search_layer(vector, curr_obj, level, points_ref);

            // Establish physical cross-link bonds with close spatial node elements
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

        // Structural scaling: raise structural bounds if node level sets a new historical peak ceiling
        if insert_level > self.max_current_level {
            self.max_current_level = insert_level;
            self.enter_node = Some(point_id);
        }

        self.nodes.insert(point_id, new_node);
    }
}
