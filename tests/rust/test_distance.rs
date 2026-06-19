// tests/rust/test_distance.rs

use fastvect::core::distance::{DistanceMetric, compute_distance};
use fastvect::core::quantization::{ScalarQuantizer, StoragePrecision};
use fastvect::{PayloadValue, Point};
use rustc_hash::FxHashMap;

/// Technical verification sequence confirming that the unstructured `Payload` state
/// behaves as a valid, polymorphic data carrier optimized for semantic cache stores.
#[test]
fn test_semantic_cache_payload_structure() {
    let mut cache_payload = FxHashMap::default();

    cache_payload.insert(
        "cached_response".to_string(),
        PayloadValue::Text(
            "{\"result\": \"Successfully generated completions API response.\"}".to_string(),
        ),
    );
    cache_payload.insert("status_code".to_string(), PayloadValue::Integer(200));

    // Wrapped raw float vector into target StoragePrecision package
    let quantized_vec = ScalarQuantizer::quantize(&[0.15, 0.22, 0.88], StoragePrecision::F32);

    let cache_point = Point {
        id: 1,
        vector: quantized_vec,
        payload: Some(cache_payload),
    };

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
    let q1 = ScalarQuantizer::quantize(&[1.0, 0.0, 0.0], StoragePrecision::F32);
    let q2 = ScalarQuantizer::quantize(&[0.0, 1.0, 0.0], StoragePrecision::F32);

    let score = compute_distance(&q1, &q2, DistanceMetric::DotProduct).unwrap();
    assert_eq!(score, 0.0);
}

/// Mathematical convergence verification confirming that perfectly aligned, identical physical directional
/// arrays produce an absolute directional similarity score of `1.0`.
#[test]
fn test_cosine_similarity_identical() {
    let q1 = ScalarQuantizer::quantize(&[3.0, 4.0, 5.0], StoragePrecision::F32);
    let q2 = ScalarQuantizer::quantize(&[3.0, 4.0, 5.0], StoragePrecision::F32);

    let similarity = compute_distance(&q1, &q2, DistanceMetric::Cosine).unwrap();
    assert!((similarity - 1.0).abs() < 1e-5);
}

/// Spatial metric check evaluating straight-line Cartesian distances using a classic 3-4-5 geometric right triangle.
#[test]
fn test_euclidean_distance_triangle() {
    let q1 = ScalarQuantizer::quantize(&[0.0, 0.0], StoragePrecision::F32);
    let q2 = ScalarQuantizer::quantize(&[3.0, 4.0], StoragePrecision::F32);

    let dist = compute_distance(&q1, &q2, DistanceMetric::HighPrecisionEuclidean).unwrap();
    assert_eq!(dist, 5.0);
}

/// Boundary security check verifying that the geometric compute layer safely isolates
/// and fends off asymmetric multi-dimensional vector inputs during calculation steps.
#[test]
fn test_dimension_mismatch_error() {
    let q1 = ScalarQuantizer::quantize(&[1.0, 2.0], StoragePrecision::F32);
    let q2 = ScalarQuantizer::quantize(&[1.0, 2.0, 3.0], StoragePrecision::F32);

    assert!(compute_distance(&q1, &q2, DistanceMetric::DotProduct).is_err());
}
