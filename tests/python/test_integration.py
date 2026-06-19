"""
Integration test suite for the fastvect high-performance vector storage engine.

Validates structural memory transitions, polymorphic payload conversions,
spatial search approximations, multi-tenancy soft isolation barriers,
new management APIs (count, delete, exists), binary persistence operations,
and multi-precision quantization modes (f32, f16, f8).
"""

import os
from collections.abc import Generator
from typing import Literal

import pytest

import fastvect


@pytest.fixture
def snapshot_cleanup() -> Generator[str, None, None]:
    """
    Pytest fixture overseeing ephemeral database lifecycle artifacts.

    Yields:
        str: The target binary snapshot file path string.
    """
    target_file = "integration_snapshot.bin"
    yield target_file
    if os.path.exists(target_file):
        os.remove(target_file)


@pytest.mark.parametrize("precision", ["f32", "f16", "f8"])
def test_vector_storage_lifecycle(
    snapshot_cleanup: str, precision: Literal["f32", "f16", "f8"]
) -> None:
    """
    Validates the end-to-end operational lifecycle of VectorStorage instances.

    Verifies programmatic upserts with heterogeneous dynamic payloads, executes
    angular cosine neighbor discovery loops, and asserts zero-loss state rehydration.
    """
    snapshot_file = snapshot_cleanup

    # Initialize core storage component with target precision layout injected dynamically
    storage = fastvect.VectorStorage(precision=precision)

    storage.upsert(
        point_id=1,
        vector=[0.1, 0.2, 0.3, 0.4],
        payload={"status": "production", "version": 1, "threshold": 0.95},
    )

    storage.upsert(
        point_id=2,
        vector=[0.9, 0.8, 0.7, 0.6],
        payload={
            "status": "staging",
            "tag": "long_description_text_" * 5,
        },
    )

    query = [0.12, 0.18, 0.32, 0.38]
    results = storage.search(query_vector=query, limit=1, metric="cosine")

    assert len(results) == 1, (
        "The search matrix must extract exactly top-1 result limit depths."
    )
    closest_id, score = results[0]
    assert closest_id == 1, (
        "Point ID 1 should mathematically optimize for minimum distance error margins."
    )
    assert score > 0.90, "Calculated cosine index metrics should map close proximity."

    storage.save(snapshot_file)
    assert os.path.exists(snapshot_file), (
        "The persistence layer failed to flush bytecode arrays to disk."
    )

    new_storage = fastvect.VectorStorage(precision=precision)
    new_storage.load(snapshot_file)

    restored_results = new_storage.search(query_vector=query, limit=1, metric="cosine")
    assert len(restored_results) == 1, (
        "The deserialized workspace partition must preserve entity indices."
    )
    assert restored_results[0][0] == 1, (
        "Analytical index continuity constraints broken following state recovery loops."
    )


@pytest.mark.parametrize("precision", ["f32", "f16", "f8"])
def test_vector_storage_single_stage_tenancy_filtration(
    precision: Literal["f32", "f16", "f8"],
) -> None:
    """
    Verifies single-stage metadata pre-filtering constraints under multi-tenant workloads.

    Ensures that vector lookups executed within shared structural graphs are safely
    restricted to the assigned tenant boundary without dropping computational search recall.
    """
    storage = fastvect.VectorStorage(precision=precision)

    storage.upsert(
        point_id=1,
        vector=[0.1, 0.1, 0.1, 0.1],
        payload={"tenant_id": "tenant_alpha", "scope": "internal"},
    )

    storage.upsert(
        point_id=2,
        vector=[0.8, 0.8, 0.8, 0.8],
        payload={"tenant_id": "tenant_beta", "scope": "external"},
    )

    query = [0.12, 0.12, 0.12, 0.12]
    results_alpha = storage.search(
        query_vector=query, limit=2, metric="cosine", tenant_id="tenant_alpha"
    )

    assert len(results_alpha) == 1, (
        "The pre-filter gatekeeper failed to drop tenant_beta completely from results."
    )
    assert results_alpha[0][0] == 1, (
        "The core search logic returned an incorrect entity key under tenant filtering fields."
    )

    results_beta = storage.search(
        query_vector=query, limit=2, metric="cosine", tenant_id="tenant_beta"
    )

    assert len(results_beta) == 1, (
        "The filtration layer allowed cross-tenant leakages during lookup operations."
    )
    assert results_beta[0][0] == 2, (
        "Analytical data routing paths cross-contaminated workspace boundaries."
    )


@pytest.mark.parametrize("precision", ["f32", "f16", "f8"])
def test_vector_storage_management_api(precision: Literal["f32", "f16", "f8"]) -> None:
    """
    Validates structural integrity boundaries for data exist, count, and delete interfaces.

    Ensures tombstone markers hide deleted entities from search paths instantly and safely
    mutates data volume status representations.
    """
    storage = fastvect.VectorStorage(precision=precision)

    assert not storage.exists(point_id=100)
    assert storage.count() == 0

    storage.upsert(
        point_id=100, vector=[0.5, 0.5, 0.5], payload={"tenant_id": "tenant_gamma"}
    )

    assert storage.exists(point_id=100)
    assert storage.count() == 1

    query = [0.5, 0.5, 0.5]
    before_delete_hits = storage.search(query_vector=query, limit=1, metric="cosine")
    assert len(before_delete_hits) == 1

    delete_success = storage.delete(point_id=100)
    assert delete_success, (
        "The deletion engine failed to process valid active record removals."
    )

    assert not storage.exists(point_id=100), (
        "Tombstone constraints failed to soft-delete the entity visibility."
    )
    assert storage.count() == 0, (
        "Global transaction counters failed to decrement metrics post-deletion."
    )

    after_delete_hits = storage.search(query_vector=query, limit=1, metric="cosine")
    assert len(after_delete_hits) == 0, (
        "Soft-deleted records leaked into active HNSW graph traversal lookups."
    )


@pytest.mark.parametrize("precision", ["f32", "f16", "f8"])
def test_vector_storage_global_and_tenant_counters(
    precision: Literal["f32", "f16", "f8"],
) -> None:
    """
    Verifies lock-free atomic transactional counter isolation properties.

    Asserts that specifying optional tenant filters correctly isolates segment
    sub-volumes without bleeding metrics into opposing environments.
    """
    storage = fastvect.VectorStorage(precision=precision)

    storage.upsert(
        point_id=10, vector=[0.1, 0.2], payload={"tenant_id": "tenant_alpha"}
    )
    storage.upsert(
        point_id=20, vector=[0.3, 0.4], payload={"tenant_id": "tenant_alpha"}
    )
    storage.upsert(point_id=30, vector=[0.5, 0.6], payload={"tenant_id": "tenant_beta"})

    assert storage.count() == 3, (
        "Global transaction counter tracked a mismatched total capacity volume."
    )
    assert storage.count(tenant_id="tenant_alpha") == 2, (
        "Tenant counter sub-indices failed to accurately parse target alpha contexts."
    )
    assert storage.count(tenant_id="tenant_beta") == 1, (
        "Tenant counter sub-indices failed to accurately parse target beta contexts."
    )
    assert storage.count(tenant_id="tenant_omega") == 0, (
        "Missing tenant contexts must fall back cleanly to a zero integer state."
    )

    _ = storage.delete(point_id=10)
    assert storage.count() == 2
    assert storage.count(tenant_id="tenant_alpha") == 1
    assert storage.count(tenant_id="tenant_beta") == 1
