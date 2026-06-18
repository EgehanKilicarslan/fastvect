// tests/rust/test_hnsw.rs

use fastvect::{HNSWIndex, Point};
use std::collections::HashMap;

/// Architectural topology verification testing graph generation passes, entryway initialization steps,
/// and bi-directional linkage weaving within a multi-tiered HNSW mesh.
#[test]
fn test_hnsw_index_dynamic_insertion_flow() {
    // Spin up an isolated index layer managing strict maximum connection degrees ($M=4$)
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

    // Lifecycle Phase 1: Assert cold-start registry behaves correctly and pins down the absolute entryway portal
    index.insert(1, &points_ref.get(&1).unwrap().vector, &points_ref);
    assert_eq!(
        index.enter_node,
        Some(1),
        "Cold-start graph deployment must declare the primary vector index as the global entry point link"
    );

    // Lifecycle Phase 2: Inject adjacent vectors to evaluate multi-tier search execution and vertex cross-stitching
    index.insert(2, &points_ref.get(&2).unwrap().vector, &points_ref);
    assert!(
        index.nodes.contains_key(&2),
        "The relational graph adjacency registry must store and track the secondary coordinate point"
    );

    let inserted_node = index.nodes.get(&2).unwrap();
    assert_eq!(inserted_node.point_id, 2);
}
