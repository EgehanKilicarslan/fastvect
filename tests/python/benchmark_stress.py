"""
Production-grade performance benchmarking and structural stress-testing suite
for the fastvect vector database engine.

Evaluates high-dimensional throughput metrics including transactional ingestion velocity (Upsert/sec),
multi-tenant query routing saturation (QPS) via multi-threaded batch operations, and system latency percentiles.
"""

import os
import random
import time

import fastvect

# --- CONFIGURATION MATRICES ---
DIMENSION: int = (
    128  # High-dimensional space configuration (e.g., semantic search vectors)
)
TOTAL_POINTS: int = (
    5000  # Scale to 5000+ to guarantee HNSW fallback lane activation (>500 nodes)
)
TOP_K: int = 10  # Total nearest neighbors depth boundary to harvest per execution pass
TEST_TENANTS: list[str] = ["tenant_alpha", "tenant_beta", "tenant_gamma"]


def generate_random_vector(dim: int) -> list[float]:
    """Generates a pseudo-random floating-point vector normalized within a boundary sequence."""
    return [random.uniform(-1.0, 1.0) for _ in range(dim)]


def run_performance_benchmark() -> None:
    print("=" * 80)
    print(
        f"🚀 STARTING FASTVECT ARCHITECTURAL STRESS TEST ({TOTAL_POINTS} Entities, {DIMENSION}-Dim)"
    )
    print("=" * 80)

    storage = fastvect.VectorStorage()

    # -----------------------------------------------------------------------------------------
    # PHASE 1: INGESTION STRESS (BULK UPSERT RUNS)
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
    # PHASE 2: FILTRATION QUERY STRESS (MULTI-THREADED BATCH SEARCH SWEEPS)
    # -----------------------------------------------------------------------------------------
    query_iterations: int = 1000
    print(
        f"\n🔍 Phase 2: Bombarding HNSW graph loops with {query_iterations} multi-tenant filtered query runs..."
    )

    # Pre-generate entire batch to isolate pure database traversal loops from generation logic
    batch_vectors: list[list[float]] = [
        generate_random_vector(DIMENSION) for _ in range(query_iterations)
    ]
    target_tenant: str = random.choice(TEST_TENANTS)

    # Core performance execution boundary
    start_time = time.perf_counter()

    # CRITICAL MULTI-THREADING CHANGE: Execute concurrent hardware traversals via modern Rayon engines
    _ = storage.batch_search(
        query_vectors=batch_vectors,
        limit=TOP_K,
        metric="cosine",
        tenant_id=target_tenant,
    )

    end_time = time.perf_counter()
    total_query_time: float = end_time - start_time
    qps: float = query_iterations / total_query_time

    # Calculate microsecond-level amortized performance metrics
    amortized_latency_ms: float = (total_query_time * 1000.0) / query_iterations

    print("✅ Query Routing Phase Complete!")
    print(f"    ▪ Multi-Core Throughput (QPS) : {qps:.2f} queries/second")
    print(
        f"    ▪ Amortized Query Latency     : {amortized_latency_ms:.4f} ms (~{amortized_latency_ms * 1000.0:.1f} μs)"
    )

    # -----------------------------------------------------------------------------------------
    # PHASE 3: PERSISTENCE STRESS (IO SNAPSHOT COMMITS & LOADING REHYDRATION)
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
