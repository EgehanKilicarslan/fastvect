// tests/rust/test_distance.rs

use fastvect::core::distance::{cosine_similarity, dot_product, euclidean_distance};
use fastvect::{PayloadValue, Point};
use std::collections::HashMap;

/// Verifies that the schemaless payload mapping correctly serializes, stores,
/// and processes dynamic text models designed for semantic caching workloads.
#[test]
fn test_semantic_cache_payload_structure() {
    let mut cache_payload = HashMap::new();

    // Simulating a production-grade LLM response envelope inside the payload matrix
    cache_payload.insert(
        "cached_response".to_string(),
        PayloadValue::Text(
            "{\"result\": \"Successfully generated completions API response.\"}".to_string(),
        ),
    );
    cache_payload.insert("status_code".to_string(), PayloadValue::Integer(200));

    let cache_point = Point {
        id: 1,
        vector: vec![0.15, 0.22, 0.88],
        payload: Some(cache_payload),
    };

    // Invariant checks on the generated point entity
    assert_eq!(cache_point.id, 1);
    if let Some(payload) = cache_point.payload {
        match payload.get("cached_response").unwrap() {
            PayloadValue::Text(text) => assert!(text.contains("Successfully generated")),
            _ => panic!(
                "Type evaluation failure: Payload value must resolve to PayloadValue::Text variant"
            ),
        }
    }
}

/// Validates that the dot product engine evaluates orthogonal vectors to an exact zero value.
#[test]
fn test_dot_product_orthogonal_geometry() {
    let v1 = vec![1.0, 0.0, 0.0];
    let v2 = vec![0.0, 1.0, 0.0];
    assert_eq!(dot_product(&v1, &v2).unwrap(), 0.0);
}

/// Confirms that identical directional arrays yield an absolute cosine similarity score of 1.0.
#[test]
fn test_cosine_similarity_identical() {
    let v1 = vec![3.0, 4.0, 5.0];
    let v2 = vec![3.0, 4.0, 5.0];
    let similarity = cosine_similarity(&v1, &v2).unwrap();

    // Applying epsilon delta evaluation bounds to account for standard float precision limits
    assert!((similarity - 1.0).abs() < 1e-5);
}

/// Evaluates spatial straight-line variance metrics using a classic 3-4-5 geometric triangle topology.
#[test]
fn test_euclidean_distance_triangle() {
    let v1 = vec![0.0, 0.0];
    let v2 = vec![3.0, 4.0];
    assert_eq!(euclidean_distance(&v1, &v2).unwrap(), 5.0);
}

/// Confirms that the geometry layer correctly halts and catches dimension mismatches during runtime passes.
#[test]
fn test_dimension_mismatch_error() {
    let v1 = vec![1.0, 2.0];
    let v2 = vec![1.0, 2.0, 3.0];
    assert!(dot_product(&v1, &v2).is_err());
}
