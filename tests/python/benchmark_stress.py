"""
Production-grade performance benchmarking and structural stress-testing suite
for the fastvect multi-precision vector database engine.

Enforces deterministic evaluation paths across F32, F16, and F8 quantized
storage layout boundaries to aggregate comparative throughput metrics.
"""

import os
import random
import time
from typing import Literal

import fastvect

# --- CONFIGURATION MATRICES ---
DIMENSION: int = (
    1536  # Industry-standard dimension (OpenAI text-embedding-3-large / Cohere v3)
)
TOTAL_POINTS: int = (
    50_000  # 50K vectors over 1536-dim forces real-world RAM-bound cache misses
)
TOP_K: int = 10  # Standard operational nearest neighbors depth
QUERY_ITERATIONS: int = (
    5000  # Lowered slightly to offset huge dimension traversal times
)
BENCHMARK_RUNS: int = 3  # Sufficient for statistical smoothing without melting the CPU
TEST_TENANTS: list[str] = [
    f"tenant_{i}" for i in range(10)
]  # Multi-tenancy noise generation (10 isolated spaces)
PRECISION_MODES: list[Literal["f32", "f16", "f8"]] = ["f32", "f16", "f8"]

# Enforce system determinism across database generation runs
random.seed(42)


def generate_random_vector(dim: int) -> list[float]:
    """Generates a pseudo-random floating-point vector normalized within boundary bounds."""
    return [random.uniform(-1.0, 1.0) for _ in range(dim)]


def run_precision_benchmark(precision: Literal["f32", "f16", "f8"]) -> None:
    print("-" * 80)
    print(
        f"🔥 EVALUATING STORAGE CONFIGURATION: Precision Mode = [{precision.upper()}]"
    )
    print("-" * 80)

    # Initialize the core storage component with target precision schema
    storage = fastvect.VectorStorage(precision=precision)

    # -----------------------------------------------------------------------------------------
    # PHASE 1: DETERMINISTIC INGESTION PERFORMANCE
    # -----------------------------------------------------------------------------------------
    print(
        f"📥 Phase 1: Injecting {TOTAL_POINTS} vectors across polymorphic tenant segments..."
    )
    start_time: float = time.perf_counter()

    for i in range(1, TOTAL_POINTS + 1):
        vec: list[float] = generate_random_vector(DIMENSION)
        assigned_tenant: str = TEST_TENANTS[i % len(TEST_TENANTS)]

        storage.upsert(
            point_id=i,
            vector=vec,
            payload={
                "tenant_id": assigned_tenant,
                "index_marker": i,
                "status": "active",
            },
        )

    end_time: float = time.perf_counter()
    total_upsert_time: float = end_time - start_time
    upsert_throughput: float = TOTAL_POINTS / total_upsert_time

    print("✅ Ingestion Phase Complete!")
    print(f"    ▪ Total Duration : {total_upsert_time:.4f} seconds")
    print(f"    ▪ Throughput     : {upsert_throughput:.2f} upserts/second")

    # -----------------------------------------------------------------------------------------
    # PHASE 2: FILTRATION QUERY STRESS WITH HARDWARE WARMUP & STATISTICAL AGGREGATION
    # -----------------------------------------------------------------------------------------
    print(
        f"\n🔍 Phase 2: Bombarding HNSW graph loops with {QUERY_ITERATIONS} filtered queries..."
    )

    batch_vectors: list[list[float]] = [
        generate_random_vector(DIMENSION) for _ in range(QUERY_ITERATIONS)
    ]
    target_tenant: str = random.choice(TEST_TENANTS)

    # HARDWARE WARMUP PHASE: Evict sleep states and map CPU cache configurations beforehand
    for _ in range(3):
        _ = storage.batch_search(
            query_vectors=batch_vectors[:50],
            limit=TOP_K,
            metric="cosine",
            tenant_id=target_tenant,
        )

    qps_records: list[float] = []
    latency_records: list[float] = []

    for _ in range(1, BENCHMARK_RUNS + 1):
        run_start = time.perf_counter()

        _ = storage.batch_search(
            query_vectors=batch_vectors,
            limit=TOP_K,
            metric="cosine",
            tenant_id=target_tenant,
        )

        run_end = time.perf_counter()
        elapsed: float = run_end - run_start

        current_qps: float = QUERY_ITERATIONS / elapsed
        current_latency_ms: float = (elapsed * 1000.0) / QUERY_ITERATIONS

        qps_records.append(current_qps)
        latency_records.append(current_latency_ms)

    avg_qps: float = sum(qps_records) / len(qps_records)
    min_qps: float = min(qps_records)
    max_qps: float = max(qps_records)
    avg_latency_ms: float = sum(latency_records) / len(latency_records)

    print("✅ Query Routing Phase Complete!")
    print(
        f"    ▪ Throughput Metrics (QPS)   : Mean: {avg_qps:.2f} | Max: {max_qps:.2f} | Min: {min_qps:.2f}"
    )
    print(
        f"    ▪ Amortized Query Latency    : {avg_latency_ms:.4f} ms (~{avg_latency_ms * 1000.0:.1f} μs)"
    )

    # -----------------------------------------------------------------------------------------
    # PHASE 3: PERSISTENCE STRESS SNAPSHOT COMMITS
    # -----------------------------------------------------------------------------------------
    snapshot_path: str = f"benchmark_stress_{precision}_snapshot.bin"
    print(
        "\n💾 Phase 3: Committing compressed zero-copy binary state serialization via Postcard..."
    )

    s_start: float = time.perf_counter()
    storage.save(snapshot_path)
    s_end: float = time.perf_counter()

    # Calculate disk footprint weight metrics
    file_size_kb: float = os.path.getsize(snapshot_path) / 1024.0
    print(f"    ▪ Serialization Save Duration : {(s_end - s_start) * 1000.0:.2f} ms")
    print(f"    ▪ Disk Snapshot Weight        : {file_size_kb:.2f} KB")

    new_storage = fastvect.VectorStorage(precision=precision)
    l_start: float = time.perf_counter()
    new_storage.load(snapshot_path)
    l_end: float = time.perf_counter()
    print(f"    ▪ Deserialization Rehydration : {(l_end - l_start) * 1000.0:.2f} ms")

    if os.path.exists(snapshot_path):
        os.remove(snapshot_path)
    print("\n")


def run_performance_suite() -> None:
    print("=" * 80)
    print("🚀 STARTING MULTI-PRECISION FASTVECT ARCHITECTURAL STRESS TEST")
    print(
        f"   Vectors: {TOTAL_POINTS} | Dimensions: {DIMENSION} | Loops: {QUERY_ITERATIONS} iterations"
    )
    print("=" * 80 + "\n")

    for precision in PRECISION_MODES:
        run_precision_benchmark(precision=precision)

    print("=" * 80)
    print("🎉 ALL FASTVECT PRECISION CONFIGURATION CORES EXECUTED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    run_performance_suite()
