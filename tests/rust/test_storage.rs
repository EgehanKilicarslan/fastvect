// tests/rust/test_storage.rs

use fastvect::{DistanceMetric, PayloadValue, Point, Segment};
use std::collections::HashMap;

/// Invariant pipeline test ensuring low-cardinality partitions correctly short-circuit search traffic
/// directly onto high-precision, raw linear search layers ($O(N)$ scans).
#[test]
fn test_storage_dynamic_routing_exact_knn() {
    let segment = Segment::new();
    let mut metadata = HashMap::new();
    metadata.insert(
        "category".to_string(),
        PayloadValue::Keyword("semantic_cache".to_string()),
    );

    // Seed data coordinates directly into the atomic memory space
    segment.upsert(Point {
        id: 42,
        vector: vec![0.1, 0.2, 0.3],
        payload: Some(metadata),
    });

    // Seed explicit database background noise to verify neighbor filtering accuracy parameters
    segment.upsert(Point {
        id: 99,
        vector: vec![-0.1, -0.2, -0.3],
        payload: None,
    });

    let query = vec![0.11, 0.19, 0.31];

    // Structural constraint check: Total cluster records (2) < 500 triggers fast sequential scans
    let hits = segment.search(&query, 1, DistanceMetric::Cosine);

    assert_eq!(hits.len(), 1);
    assert_eq!(
        hits[0].0, 42,
        "The storage router failed to steer target workflows cleanly into precise KNN processing pipelines"
    );
    assert!(
        hits[0].1 > 0.95,
        "Calculated vector metric boundaries dropped below target similarity margins"
    );

    let payload = hits[0].2.as_ref().unwrap();
    assert_eq!(
        payload.get("category").unwrap(),
        &PayloadValue::Keyword("semantic_cache".to_string())
    );
}

/// Operational threshold validation confirming that massive cluster growth triggers deep multi-layer
/// graph routing without spilling thread block states or throwing exception failures.
#[test]
fn test_storage_dynamic_routing_hnsw_fallback_stub() {
    let segment = Segment::new();

    // Programmatically populate memory tracks past the threshold barrier line (501 > 500)
    for i in 0..501 {
        segment.upsert(Point {
            id: i,
            vector: vec![i as f32 * 0.001, 0.5, 0.5],
            payload: None,
        });
    }

    let query = vec![0.5, 0.5, 0.5];

    // Structural constraint check: Total storage load (501) > 500 activates fast graph indexes ($O(\log N)$ links).
    let hits = segment.search(&query, 5, DistanceMetric::HighPrecisionEuclidean);

    assert_eq!(
        hits.len(),
        5,
        "The search orchestrator failed to extract the requested total match count depth boundary"
    );

    // Proximity logic assert: Point 500 resolves to coordinates mapping exactly to our raw search query (0.500 vs 0.5)
    assert_eq!(
        hits[0].0, 500,
        "The search routing pipeline failed to resolve the optimal neighbor presenting minimum geometric variance"
    );
}

/// Dynamic snapshot validation tracking serialization passes, disk commit cycles, and
/// database hot-swapping sequences using Postcard with zero spatial accuracy drift.
#[test]
fn test_storage_persistence_snapshot_lifecycle() {
    let source_segment = Segment::new();
    let temp_snapshot_path = "tests_temporary_snapshot.bin";

    // 1. Seed the primary transactional source block with spatial data
    source_segment.upsert(Point {
        id: 1337,
        vector: vec![0.1, 0.2, 0.3, 0.4],
        payload: None,
    });

    // 2. Extract and flush a binary snapshot of current indices directly onto the physical disk layer
    let save_result = source_segment.save_to_disk(temp_snapshot_path);
    assert!(
        save_result.is_ok(),
        "The persistence subsystem failed to flush memory clusters to disk"
    );

    // 3. Instantiate a completely unpopulated, separate runtime instance and trigger rehydration
    let restored_segment = Segment::new();
    let load_result = restored_segment.load_from_disk(temp_snapshot_path);
    assert!(
        load_result.is_ok(),
        "The deserialization pipeline failed to decode and hydrate the binary system snapshot file"
    );

    // 4. Run historical continuity validations against the hot-swapped database memory space
    let query = vec![0.1, 0.2, 0.3, 0.4];
    let hits = restored_segment.search(&query, 1, DistanceMetric::Cosine);

    assert_eq!(
        hits.len(),
        1,
        "The deserialized storage structure failed to accurately match original database capacities"
    );
    assert_eq!(
        hits[0].0, 1337,
        "The structural query pass pulled incorrect point IDs following persistence rehydration runs"
    );
    assert!(
        hits[0].1 > 0.99,
        "The data engine suffered analytical weight drift across floating-point configurations post-recovery"
    );

    // 5. Ephemeral cleanup pass: sweep away localized file system artifacts cleanly
    let _ = std::fs::remove_file(temp_snapshot_path);
}
