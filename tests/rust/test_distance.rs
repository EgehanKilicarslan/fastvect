// tests/rust/test_distance.rs

use fastvect::core::distance::{cosine_similarity, dot_product, euclidean_distance};
use fastvect::{PayloadValue, Point};
use std::collections::HashMap;

/// Technical verification sequence confirming that the unstructured `Payload` state
/// behaves as a valid, polymorphic data carrier optimized for semantic cache stores.
#[test]
fn test_semantic_cache_payload_structure() {
    let mut cache_payload = HashMap::new();

    // Mocking an operational JSON completion response payload inside our variant mapping
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

    // Assert structural schema and payload integrity constraints
    assert_eq!(cache_point.id, 1);
    if let Some(payload) = cache_point.payload {
        match payload.get("cached_response").unwrap() {
            PayloadValue::Text(text) => assert!(text.contains("Successfully generated")),
            _ => panic!(
                "Type assertion error: Payload data must resolve exactly to PayloadValue::Text"
            ),
        }
    }
}

/// Geometric invariant test ensuring that completely orthogonal high-dimensional vectors
/// yield a scalar dot product score of exactly `0.0`.
#[test]
fn test_dot_product_orthogonal_geometry() {
    let v1 = vec![1.0, 0.0, 0.0];
    let v2 = vec![0.0, 1.0, 0.0];
    assert_eq!(dot_product(&v1, &v2).unwrap(), 0.0);
}

/// Mathematical convergence verification confirming that perfectly aligned, identical physical directional
/// arrays produce an absolute directional similarity score of `1.0`.
#[test]
fn test_cosine_similarity_identical() {
    let v1 = vec![3.0, 4.0, 5.0];
    let v2 = vec![3.0, 4.0, 5.0];
    let similarity = cosine_similarity(&v1, &v2).unwrap();

    // Integrate an analytical epsilon bound to safely bypass dynamic floating-point rounding errors
    assert!((similarity - 1.0).abs() < 1e-5);
}

/// Spatial metric check evaluating straight-line Cartesian distances using a classic 3-4-5 geometric right triangle.
#[test]
fn test_euclidean_distance_triangle() {
    let v1 = vec![0.0, 0.0];
    let v2 = vec![3.0, 4.0];
    assert_eq!(euclidean_distance(&v1, &v2).unwrap(), 5.0);
}

/// Boundary security check verifying that the geometric compute layer safely isolates
/// and fends off asymmetric multi-dimensional vector inputs during calculation steps.
#[test]
fn test_dimension_mismatch_error() {
    let v1 = vec![1.0, 2.0];
    let v2 = vec![1.0, 2.0, 3.0];
    assert!(dot_product(&v1, &v2).is_err());
}
