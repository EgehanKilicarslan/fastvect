// tests/rust/test_hnsw.rs

use fastvect::{HNSWIndex, Point};
use std::collections::HashMap;

/// Validates cold-start instantiation routines, node linkage transitions,
/// and entrance matrix scaling within the multi-layer HNSW graph framework.
#[test]
fn test_hnsw_index_dynamic_insertion_flow() {
    // Instantiate an independent graph partition index structure with aggressive connection bounds
    let mut index = HNSWIndex::new(4, 16, 16);
    let mut points_ref = HashMap::new();

    let p1 = Point {
        id: 1,
        vector: vec![1.0, 0.0],
        payload: None,
    };
    let p2 = Point {
        id: 2,
        vector: vec![0.0, 1.0],
        payload: None,
    };

    points_ref.insert(1, p1);
    points_ref.insert(2, p2);

    // Operational Phase 1: Test entry anchor generation on initial system mutation
    index.insert(1, &points_ref.get(&1).unwrap().vector, &points_ref);
    assert_eq!(
        index.enter_node,
        Some(1),
        "Initial node insertion must declare the global entry point anchor"
    );

    // Operational Phase 2: Insert a secondary vector coordinate to trigger multi-tier routing matrices
    index.insert(2, &points_ref.get(&2).unwrap().vector, &points_ref);
    assert!(
        index.nodes.contains_key(&2),
        "Target graph layout register must index the secondary vertex"
    );

    let inserted_node = index.nodes.get(&2).unwrap();
    assert_eq!(inserted_node.point_id, 2);
}
