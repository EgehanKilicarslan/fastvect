"""
Integration test suite for the fastvect high-performance vector storage engine.

This module validates structural memory transitions, polymorphic payload conversions,
spatial search approximations, multi-tenancy soft isolation barriers, and binary
persistence operations using pytest assertions.
"""

import os
from collections.abc import Generator

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
    # Teardown phase: Safely sweep disk layers regardless of execution success
    if os.path.exists(target_file):
        os.remove(target_file)


def test_vector_storage_lifecycle(snapshot_cleanup: str) -> None:
    """
    Validates the end-to-end operational lifecycle of VectorStorage instances.

    Verifies programmatic upserts with heterogeneous dynamic payloads, executes
    angular cosine neighbor discovery loops, and asserts zero-loss state rehydration.

    Args:
        snapshot_cleanup (str): Fixture injected path targeting file persistence checks.
    """
    snapshot_file = snapshot_cleanup

    # Initialize the core storage component wrapped from Rust
    storage = fastvect.VectorStorage()

    # 1. Pipeline Verification: Upsert entries with structured metadata matrices
    storage.upsert(
        point_id=1,
        vector=[0.1, 0.2, 0.3, 0.4],
        payload={"status": "production", "version": 1, "threshold": 0.95},
    )

    # Intentionally trigger the Text variant by pushing a string exceeding 64 characters
    storage.upsert(
        point_id=2,
        vector=[0.9, 0.8, 0.7, 0.6],
        payload={
            "status": "staging",
            "tag": "long_description_text_" * 5,
        },
    )

    # 2. Match Validation: Run cosine spatial evaluations near targeted clusters
    query = [0.12, 0.18, 0.32, 0.38]
    results = storage.search(query_vector=query, limit=1, metric="cosine")

    assert len(results) == 1, (
        "The search matrix must extract exactly top-1 result limit depths."
    )
    closest_id, score = results[0]
    assert closest_id == 1, (
        "Point ID 1 should mathematically optimize for minimum distance error margins."
    )
    assert score > 0.95, "Calculated cosine index metrics should map close proximity."

    # 3. Continuity Verification: Save snapshot and rehydrate from an isolated instance
    storage.save(snapshot_file)
    assert os.path.exists(snapshot_file), (
        "The persistence layer failed to flush bytecode arrays to disk."
    )

    new_storage = fastvect.VectorStorage()
    new_storage.load(snapshot_file)

    restored_results = new_storage.search(query_vector=query, limit=1, metric="cosine")
    assert len(restored_results) == 1, (
        "The deserialized workspace partition must preserve entity indices."
    )
    assert restored_results[0][0] == 1, (
        "Analytical index continuity constraints broken following state recovery loops."
    )


def test_vector_storage_single_stage_tenancy_filtration() -> None:
    """
    Verifies single-stage metadata pre-filtering constraints under multi-tenant workloads.

    Ensures that vector lookups executed within shared structural graphs are safely
    restricted to the assigned tenant boundary without dropping computational search recall.
    """
    storage = fastvect.VectorStorage()

    # 1. Seed workspace partitions separating tenant_alpha from tenant_beta
    # Point 1 is mathematically closest to the query but explicitly tied to tenant_alpha
    storage.upsert(
        point_id=1,
        vector=[0.1, 0.1, 0.1, 0.1],
        payload={"tenant_id": "tenant_alpha", "scope": "internal"},
    )

    # Point 2 sits further away and is explicitly locked under tenant_beta properties
    storage.upsert(
        point_id=2,
        vector=[0.8, 0.8, 0.8, 0.8],
        payload={"tenant_id": "tenant_beta", "scope": "external"},
    )

    query = [0.12, 0.12, 0.12, 0.12]

    # 2. Verify Tenancy Gatekeeper: Execute spatial search restricted to tenant_alpha boundaries
    results_alpha = storage.search(
        query_vector=query, limit=2, metric="cosine", tenant_id="tenant_alpha"
    )

    assert len(results_alpha) == 1, (
        "The pre-filter gatekeeper failed to drop tenant_beta completely from results."
    )
    assert results_alpha[0][0] == 1, (
        "The core search logic returned an incorrect entity key under tenant filtering fields."
    )

    # 3. Reverse Boundaries: Execute spatial search restricted to tenant_beta boundaries
    results_beta = storage.search(
        query_vector=query, limit=2, metric="cosine", tenant_id="tenant_beta"
    )

    assert len(results_beta) == 1, (
        "The filtration layer allowed cross-tenant leakages during lookup operations."
    )
    assert results_beta[0][0] == 2, (
        "Analytical data routing paths cross-contaminated workspace boundaries."
    )
