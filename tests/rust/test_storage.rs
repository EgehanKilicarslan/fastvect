// tests/rust/test_storage.rs

use fastvect::{DistanceMetric, PayloadValue, Point, Segment};
use std::collections::HashMap;

/// Verifies that low-capacity segments correctly route request parameters
/// directly to high-precision exact KNN execution lines.
#[test]
fn test_storage_dynamic_routing_exact_knn() {
    let segment = Segment::new();
    let mut metadata = HashMap::new();
    metadata.insert(
        "category".to_string(),
        PayloadValue::Keyword("semantic_cache".to_string()),
    );

    // Upsert target coordinate profiles directly into the operational partition mesh
    segment.upsert(Point {
        id: 42,
        vector: vec![0.1, 0.2, 0.3],
        payload: Some(metadata),
    });

    // Upsert semantic noise entry vectors to confirm clustering accuracy
    segment.upsert(Point {
        id: 99,
        vector: vec![-0.1, -0.2, -0.3],
        payload: None,
    });

    let query = vec![0.11, 0.19, 0.31];

    // Execution routing invariant: total points (2) < 500 triggers precise linear sweeps
    let hits = segment.search(&query, 1, DistanceMetric::Cosine);

    assert_eq!(hits.len(), 1);
    assert_eq!(
        hits[0].0, 42,
        "The evaluation engine must route queries cleanly to exact matching nodes"
    );
    assert!(
        hits[0].1 > 0.95,
        "Calculated cosine similarity value profile should be high"
    );

    let payload = hits[0].2.as_ref().unwrap();
    assert_eq!(
        payload.get("category").unwrap(),
        &PayloadValue::Keyword("semantic_cache".to_string())
    );
}

/// Assures that large scaling bounds force the orchestrator past standard capacity thresholds,
/// validating stable traversal cascades across deep HNSW network paths.
#[test]
fn test_storage_dynamic_routing_hnsw_fallback_stub() {
    let segment = Segment::new();

    // Rapidly populate data buffers to force the system layout state machine past the 500 limit boundary
    for i in 0..501 {
        segment.upsert(Point {
            id: i,
            vector: vec![i as f32 * 0.001, 0.5, 0.5],
            payload: None,
        });
    }

    let query = vec![0.5, 0.5, 0.5];

    // Execution routing invariant: cluster allocation counts (501) > 500 activates deep graphical indexes.
    // The underlying fallback interface must resolve queries correctly without encountering critical thread runtime block errors.
    let hits = segment.search(&query, 5, DistanceMetric::HighPrecisionEuclidean);

    assert_eq!(
        hits.len(),
        5,
        "The structural query pass should successfully extract the requested top-K match depth bounds"
    );

    // Cartesian minimization rule: lower absolute Euclidean variance mappings verify the highest spatial optimization parameters.
    // Index position 500 contains coordinates aligned precisely with the target parameters array (0.500 vs 0.5).
    assert_eq!(
        hits[0].0, 500,
        "The computational layout matrix must properly prioritize indices presenting the lowest geometric margin errors"
    );
}
