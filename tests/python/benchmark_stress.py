"""
Production-grade performance benchmarking and structural stress-testing suite
for the fastvect vector database engine.

Optimized to enforce deterministic evaluation paths, isolate execution warmup boundaries,
and aggregate statistical data points across multi-threaded operations.
"""

import os
import random
import time

import fastvect

# --- CONFIGURATION MATRICES ---
DIMENSION: int = 128  # High-dimensional embedding configuration space
TOTAL_POINTS: int = (
    5000  # Guarantees HNSW hierarchical lane fallback trigger (>500 nodes)
)
TOP_K: int = 10  # Total nearest neighbors depth threshold (Top-K results)
QUERY_ITERATIONS: int = 1000  # Total queries sent inside the hardware traversal loops
BENCHMARK_RUNS: int = (
    5  # Statistical iterations to isolate thermal and OS throttling anomalies
)
TEST_TENANTS: list[str] = ["tenant_alpha", "tenant_beta", "tenant_gamma"]

# Enforce system determinism across database generation runs
random.seed(42)


def generate_random_vector(dim: int) -> list[float]:
    """Generates a pseudo-random floating-point vector normalized within boundary bounds."""
    return [random.uniform(-1.0, 1.0) for _ in range(dim)]


def run_performance_benchmark() -> None:
    print("=" * 80)
    print(
        f"🚀 STARTING FASTVECT ARCHITECTURAL STRESS TEST ({TOTAL_POINTS} Entities, {DIMENSION}-Dim)"
    )
    print("=" * 80)

    storage = fastvect.VectorStorage()

    # -----------------------------------------------------------------------------------------
    # PHASE 1: DETERMINISTIC INGESTION PERFORMANCE
    # -----------------------------------------------------------------------------------------
    print(
        f"\n📥 Phase 1: Injecting {TOTAL_POINTS} vectors across polymorphic tenant segments..."
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
        f"\n🔍 Phase 2: Bombarding HNSW graph loops with {QUERY_ITERATIONS} multi-tenant filtered queries..."
    )

    # Pre-generate entire batch to isolate pure database traversal from python loop allocations
    batch_vectors: list[list[float]] = [
        generate_random_vector(DIMENSION) for _ in range(QUERY_ITERATIONS)
    ]
    target_tenant: str = random.choice(TEST_TENANTS)

    # HARDWARE WARMUP PHASE: Evict sleep states and map CPU cache configurations beforehand
    print("    🔥 Executing query loops warmup boundary context pass...")
    for _ in range(3):
        _ = storage.batch_search(
            query_vectors=batch_vectors[:50],
            limit=TOP_K,
            metric="cosine",
            tenant_id=target_tenant,
        )

    # Core performance metric tracking containers
    qps_records: list[float] = []
    latency_records: list[float] = []

    print(
        f"    ⏱️  Running {BENCHMARK_RUNS} benchmark cycles for statistical stability profiling..."
    )
    for run in range(1, BENCHMARK_RUNS + 1):
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
        print(
            f"        ▪ Cycle #{run}: {current_qps:.2f} QPS | Latency: {current_latency_ms * 1000.0:.1f} μs"
        )

    # Extract clean statistical summary representations
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
    snapshot_path: str = "benchmark_stress_snapshot.bin"
    print(
        "\n💾 Phase 3: Committing compressed zero-copy binary state serialization via Postcard..."
    )

    s_start: float = time.perf_counter()
    storage.save(snapshot_path)
    s_end: float = time.perf_counter()
    print(f"    ▪ Serialization Save Duration : {(s_end - s_start) * 1000.0:.2f} ms")

    new_storage: fastvect.VectorStorage = fastvect.VectorStorage()
    l_start: float = time.perf_counter()
    new_storage.load(snapshot_path)
    l_end: float = time.perf_counter()
    print(f"    ▪ Deserialization Rehydration : {(l_end - l_start) * 1000.0:.2f} ms")

    if os.path.exists(snapshot_path):
        os.remove(snapshot_path)

    print("\n" + "=" * 80)
    print("🎉 FASTVECT ARCHITECTURAL PERFORMANCE BENCHMARK EXECUTED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    run_performance_benchmark()
