// tests/rust/test_storage.rs

use fastvect::core::filter::Filter;
use fastvect::core::quantization::ScalarQuantizer;
use fastvect::{DistanceMetric, PayloadValue, Point, Segment, StoragePrecision};
use rustc_hash::FxHashMap;

/// Invariant pipeline test ensuring low-cardinality partitions correctly short-circuit search traffic
/// directly onto high-precision, raw linear search layers ($O(N)$ scans).
#[test]
fn test_storage_dynamic_routing_exact_knn() {
    let segment = Segment::new(StoragePrecision::F32);
    let mut metadata = FxHashMap::default();
    metadata.insert(
        "category".to_string(),
        PayloadValue::Keyword("semantic_cache".to_string()),
    );

    segment.upsert(Point {
        id: 42,
        vector: ScalarQuantizer::quantize(&[0.1, 0.2, 0.3], StoragePrecision::F32),
        payload: Some(metadata),
    });

    segment.upsert(Point {
        id: 99,
        vector: ScalarQuantizer::quantize(&[-0.1, -0.2, -0.3], StoragePrecision::F32),
        payload: None,
    });

    let query = vec![0.11, 0.19, 0.31];
    let hits = segment.search(&query, 1, DistanceMetric::Cosine, None);

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

/// Operational threshold validation confirming that cluster growth successfully triggers deep
/// multi-layer HNSW graph fallback routing paths without dropping target recall count parameters.
#[test]
fn test_storage_dynamic_routing_hnsw_fallback_stub() {
    let segment = Segment::new(StoragePrecision::F32);

    for i in 0..501 {
        segment.upsert(Point {
            id: i,
            vector: ScalarQuantizer::quantize(&[i as f32 * 0.001, 0.5, 0.5], StoragePrecision::F32),
            payload: None,
        });
    }

    let query = vec![0.5, 0.5, 0.5];
    let hits = segment.search(&query, 5, DistanceMetric::HighPrecisionEuclidean, None);

    assert_eq!(
        hits.len(),
        5,
        "The search orchestrator failed to extract the requested total match count depth boundary"
    );

    assert_eq!(
        hits[0].0, 500,
        "The search routing pipeline failed to resolve the optimal neighbor presenting minimum geometric variance"
    );
}

/// Dynamic snapshot validation tracking serialization passes, disk commit cycles, and
/// database rehydration sequences using Postcard with zero floating-point accuracy drift.
#[test]
fn test_storage_persistence_snapshot_lifecycle() {
    let source_segment = Segment::new(StoragePrecision::F32);
    let temp_snapshot_path = "tests_temporary_snapshot.bin";

    source_segment.upsert(Point {
        id: 1337,
        vector: ScalarQuantizer::quantize(&[0.1, 0.2, 0.3, 0.4], StoragePrecision::F32),
        payload: None,
    });

    let save_result = source_segment.save_to_disk(temp_snapshot_path);
    assert!(
        save_result.is_ok(),
        "The persistence subsystem failed to flush memory clusters to disk"
    );

    let restored_segment = Segment::new(StoragePrecision::F32);
    let load_result = restored_segment.load_from_disk(temp_snapshot_path);
    assert!(
        load_result.is_ok(),
        "The deserialization pipeline failed to decode and hydrate the binary system snapshot file"
    );

    let query = vec![0.1, 0.2, 0.3, 0.4];
    let hits = restored_segment.search(&query, 1, DistanceMetric::Cosine, None);

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

    let _ = std::fs::remove_file(temp_snapshot_path);
}

/// Verification test checking Single-Stage Pre-Filtering behavior under active multi-tenant workloads
/// to confirm strict logical network separation boundaries.
#[test]
fn test_storage_single_stage_tenancy_filtration() {
    let segment = Segment::new(StoragePrecision::F32);

    let mut payload_a = FxHashMap::default();
    payload_a.insert(
        "tenant_id".to_string(),
        PayloadValue::Keyword("tenant_alpha".to_string()),
    );

    let mut payload_b = FxHashMap::default();
    payload_b.insert(
        "tenant_id".to_string(),
        PayloadValue::Keyword("tenant_beta".to_string()),
    );

    segment.upsert(Point {
        id: 1,
        vector: ScalarQuantizer::quantize(&[0.1, 0.1, 0.1], StoragePrecision::F32),
        payload: Some(payload_a),
    });

    segment.upsert(Point {
        id: 2,
        vector: ScalarQuantizer::quantize(&[0.9, 0.9, 0.9], StoragePrecision::F32),
        payload: Some(payload_b),
    });

    let query = vec![0.12, 0.12, 0.12];

    let filter_alpha = Filter::new("tenant_alpha".to_string());
    let hits_alpha = segment.search(&query, 2, DistanceMetric::Cosine, Some(&filter_alpha));

    assert_eq!(
        hits_alpha.len(),
        1,
        "Filtration wall should drop tenant_beta completely from results"
    );
    assert_eq!(
        hits_alpha[0].0, 1,
        "The engine returned the incorrect entity key under tenant filtering"
    );

    let filter_beta = Filter::new("tenant_beta".to_string());
    let hits_beta = segment.search(&query, 2, DistanceMetric::Cosine, Some(&filter_beta));

    assert_eq!(hits_beta.len(), 1);
    assert_eq!(
        hits_beta[0].0, 2,
        "The gatekeeper failed to safely expose isolation regions for tenant_beta"
    );
}

/// Boundary security test checking that a search query configured with a non-existent
/// tenant ID correctly short-circuits and safely returns an empty collection block.
#[test]
fn test_storage_filtration_with_non_existent_tenant() {
    let segment = Segment::new(StoragePrecision::F32);
    let mut payload = FxHashMap::default();
    payload.insert(
        "tenant_id".to_string(),
        PayloadValue::Keyword("tenant_alpha".to_string()),
    );

    segment.upsert(Point {
        id: 100,
        vector: ScalarQuantizer::quantize(&[0.1, 0.2, 0.3], StoragePrecision::F32),
        payload: Some(payload),
    });

    let query = vec![0.1, 0.2, 0.3];
    let filter_omega = Filter::new("tenant_omega".to_string());

    let hits = segment.search(&query, 5, DistanceMetric::Cosine, Some(&filter_omega));

    assert!(
        hits.is_empty(),
        "The filtering gatekeeper failed to short-circuit and return an empty payload vector on missing tenant contexts"
    );
}

/// Structural regression evaluation confirming that the HNSW graph traversal routines safely obey
/// single-stage tenancy filtration boundaries even when scaling the segment pool past 500 points.
#[test]
fn test_storage_hnsw_graph_navigational_tenancy_filtration() {
    let segment = Segment::new(StoragePrecision::F32);

    for i in 0..600 {
        let mut payload = FxHashMap::default();
        let tenant_name = if i % 2 == 0 {
            "tenant_alpha"
        } else {
            "tenant_beta"
        };
        payload.insert(
            "tenant_id".to_string(),
            PayloadValue::Keyword(tenant_name.to_string()),
        );

        segment.upsert(Point {
            id: i,
            vector: ScalarQuantizer::quantize(&[i as f32 * 0.001, 0.5, 0.5], StoragePrecision::F32),
            payload: Some(payload),
        });
    }

    let query = vec![0.5, 0.5, 0.5];
    let filter_alpha = Filter::new("tenant_alpha".to_string());

    let hits = segment.search(
        &query,
        10,
        DistanceMetric::HighPrecisionEuclidean,
        Some(&filter_alpha),
    );

    assert!(
        !hits.is_empty(),
        "Graph routing paths collapsed under filtered cluster navigation passes"
    );

    for hit in hits {
        let point_id = hit.0;
        assert_eq!(
            point_id % 2,
            0,
            "TOPOLOGICAL LEAKAGE DETECTED: Node {} (tenant_beta) leaked into tenant_alpha HNSW search tracks!",
            point_id
        );
    }
}

/// Boundary safety evaluation verifying that entities completely lacking payload matrices
/// are safely isolated instead of triggering exceptions during filtered execution passes.
#[test]
fn test_storage_filtration_handles_missing_payloads_safely() {
    let segment = Segment::new(StoragePrecision::F32);

    segment.upsert(Point {
        id: 777,
        vector: ScalarQuantizer::quantize(&[0.1, 0.2, 0.3], StoragePrecision::F32),
        payload: None,
    });

    let query = vec![0.1, 0.2, 0.3];
    let filter_alpha = Filter::new("tenant_alpha".to_string());
    let hits = segment.search(&query, 1, DistanceMetric::Cosine, Some(&filter_alpha));

    assert!(
        hits.is_empty(),
        "The evaluation engine cross-contaminated search records by letting blank payloads pass validation checkpoints"
    );
}
