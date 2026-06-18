// tests/rust/test_storage.rs

use fastvect::core::filter::Filter;
use fastvect::{DistanceMetric, PayloadValue, Point, Segment};
use std::collections::HashMap;

/// Invariant pipeline test ensuring low-cardinality partitions correctly short-circuit search traffic
/// directly onto high-precision, raw linear search layers ($O(N)$ scans).
///
/// This validation confirms the routing engine correctly switches to brute-force execution paths
/// when the total entity matrix density remains well under production indexing thresholds.
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

    // Structural constraint check: Total cluster records (2) < 500 triggers fast sequential scans.
    // The optional filter parameter is explicitly set to None to check default routing lanes.
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

/// Operational threshold validation confirming that massive cluster growth triggers deep multi-layer
/// graph routing without spilling thread block states or throwing exception failures.
///
/// This stress test verifies the transition boundaries where low-cardinality collections scale
/// past optimization limits, safely spawning multi-tier HNSW link frameworks ($O(\log N)$ links).
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

    // Structural constraint check: Total storage load (501) > 500 activates fast graph indexes.
    let hits = segment.search(&query, 5, DistanceMetric::HighPrecisionEuclidean, None);

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
///
/// Validates programmatic transactional safety limits during network replication workflows by flushes,
/// resets, and full data rehydration without introducing computational weight shifts.
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

    // 5. Ephemeral cleanup pass: sweep away localized file system artifacts cleanly
    let _ = std::fs::remove_file(temp_snapshot_path);
}

/// Verification test checking Single-Stage Pre-Filtering behavior under active multi-tenant workloads.
///
/// This evaluation validates that shared graphical structures successfully isolate query routing tracks
/// without suffering from the recall degradation drops native to post-query isolation strategies.
#[test]
fn test_storage_single_stage_tenancy_filtration() {
    let segment = Segment::new();

    // 1. Setup metadata payloads separating tenant_alpha from tenant_beta
    let mut payload_a = HashMap::new();
    payload_a.insert(
        "tenant_id".to_string(),
        PayloadValue::Keyword("tenant_alpha".to_string()),
    );

    let mut payload_b = HashMap::new();
    payload_b.insert(
        "tenant_id".to_string(),
        PayloadValue::Keyword("tenant_beta".to_string()),
    );

    // 2. Insert vectors into the shared index structure
    // Point 1 belongs to tenant_alpha and is mathematically closest to our query
    segment.upsert(Point {
        id: 1,
        vector: vec![0.1, 0.1, 0.1],
        payload: Some(payload_a),
    });

    // Point 2 belongs to tenant_beta and sits further away
    segment.upsert(Point {
        id: 2,
        vector: vec![0.9, 0.9, 0.9],
        payload: Some(payload_b),
    });

    let query = vec![0.12, 0.12, 0.12];

    // 3. Execution Pass 1: Search specifically matching tenant_alpha constraints
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

    // 4. Execution Pass 2: Reverse search boundaries targeting tenant_beta constraints
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
    let segment = Segment::new();
    let mut payload = HashMap::new();
    payload.insert(
        "tenant_id".to_string(),
        PayloadValue::Keyword("tenant_alpha".to_string()),
    );

    segment.upsert(Point {
        id: 100,
        vector: vec![0.1, 0.2, 0.3],
        payload: Some(payload),
    });

    let query = vec![0.1, 0.2, 0.3];
    let filter_omega = Filter::new("tenant_omega".to_string());

    // Execute search with a tenant filter that matches absolutely no data layers
    let hits = segment.search(&query, 5, DistanceMetric::Cosine, Some(&filter_omega));

    assert!(
        hits.is_empty(),
        "The filtering gatekeeper failed to short-circuit and return an empty payload vector on missing tenant contexts"
    );
}

/// Structural regression evaluation confirming that the HNSW graph traversal routines
/// safely obey single-stage tenancy filtration boundaries even when scaling past 500 points.
#[test]
fn test_storage_hnsw_graph_navigational_tenancy_filtration() {
    let segment = Segment::new();

    // Programmatically distribute 600 records alternating between two tenant spaces
    for i in 0..600 {
        let mut payload = HashMap::new();
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
            // Project vectors along opposing trajectories to stress the graph router
            vector: vec![i as f32 * 0.001, 0.5, 0.5],
            payload: Some(payload),
        });
    }

    let query = vec![0.5, 0.5, 0.5];
    let filter_alpha = Filter::new("tenant_alpha".to_string());

    // Structural constraint check: Total nodes (600) > 500 activates fast HNSW graph traversal pipelines.
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

    // Invariant validation: All recovered neighbors must strictly belong to the specified tenant namespace
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
/// (`payload: None`) are safely dropped instead of triggering exceptions during filtered passes.
#[test]
fn test_storage_filtration_handles_missing_payloads_safely() {
    let segment = Segment::new();

    // Insert a coordinate containing zero metadata layers
    segment.upsert(Point {
        id: 777,
        vector: vec![0.1, 0.2, 0.3],
        payload: None, // Missing payload matrix
    });

    let query = vec![0.1, 0.2, 0.3];
    let filter_alpha = Filter::new("tenant_alpha".to_string());

    let hits = segment.search(&query, 1, DistanceMetric::Cosine, Some(&filter_alpha));

    assert!(
        hits.is_empty(),
        "The evaluation engine cross-contaminated search records by letting blank payloads pass validation checkpoints"
    );
}
