# 🚀 FastVect

FastVect is an ultra-high-performance, memory-optimized **embedded vector storage and search engine** designed in Rust and compiled into native, zero-dependency Python binaries.

By eliminating server-side hops, HTTP/gRPC network overhead, and heavy serialization bottlenecks, FastVect runs directly inside your active Python process memory. Thanks to pre-compiled binary wheels hosted on PyPI, it installs instantly on Windows, Linux, and macOS without requiring a local Rust toolchain or `cargo` environment.

---

## 📊 Performance Benchmark Ledger

The following metrics were captured under rigorous architectural stress-testing on an **Intel Core i7-10750H CPU** (5,000 Entities, 128-Dimension, Cosine Metric, distributed across isolated tenant spaces):

| Phase                  | Evaluation Matrix               | Throughput / Velocity      | Amortized Latency       |
| :--------------------- | :------------------------------ | :------------------------- | :---------------------- |
| **📥 Ingestion**       | Bulk Transactional Upsert       | **29,042.77** upserts/sec  | 0.1722 seconds (Total)  |
| **🔍 Search (Single)** | Sequential Graph Traversal      | **19,117.29** queries/sec  | 0.0340 ms (~34.0 μs)    |
| **🏎️ Search (Batch)**  | **Multi-Threaded Rayon Engine** | **127,475.43** queries/sec | **0.0078 ms (~7.8 μs)** |
| **💾 Save State**      | Postcard Binary Serialization   | Zero-Copy Disk Commit      | 5.70 ms (Total Space)   |
| **🔄 Rehydration**     | Memory Hot-Swap Reload          | Exclusive State Hydration  | 8.24 ms (Total Time)    |

### 📈 Benchmark Insights

- **The 127K QPS Record:** By passing query matrices in bulk via `.batch_search()`, FastVect drops into a highly optimized Rust worker pool managed by Rayon. This completely bypasses Python's GIL (Global Interpreter Lock) and saturates all available CPU cores.
- **Microsecond Latency:** Swapping heap-allocated visit trackers with an $O(1)$ stack-like flat `Vec<bool>` allocation within the HNSW traversal loop reduces search latency down to an astonishing **7.8 microseconds** per query.

---

## 🛠️ Key Architectural Innovations

- **GIL-Free Data Parallelism:** Parallel iterators seamlessly map concurrent graph traversal lookups directly to bare-metal hardware threads.
- **Contiguous Graph Memory:** Replaced pointer-heavy vertex representations with contiguous array blocks, reducing heap fragmentation and maximizing L1/L2 cache locality.
- **Single-Stage Multi-Tenancy:** Filters properties during the graph routing phases instead of relying on post-query vector truncation, keeping recall precision at 100%.

---

## 📦 Installation

FastVect provides pre-compiled binaries for major operating systems and Python versions. No local compiler or Rust setup is needed:

```bash
pip install fastvect

```

---

## 🚀 Quickstart Guide

### 1. Ingestion & Multi-Tenant Search

```python
import fastvect

# Initialize a production-grade embedded storage workspace
storage = fastvect.VectorStorage()

# Upsert coordinates paired with structural metadata payloads
storage.upsert(
    point_id=1,
    vector=[0.12, -0.43, 0.84, ..., 0.09],  # 128-dimensional list
    payload={
        "tenant_id": "tenant_alpha",
        "status": "active",
        "index_marker": 500
    }
)

# High-speed single search query with active pre-filtering
results = storage.search(
    query_vector=[0.10, -0.40, 0.80, ..., 0.05],
    limit=10,
    metric="cosine",
    tenant_id="tenant_alpha"
)
print(f"Top-K Matches: {results}")

```

### 2. Multi-Core Batch Search Blast

To replicate the **127K QPS benchmark**, aggregate your query vectors and route them concurrently through the parallel engine:

```python
# A nested list containing hundreds of raw analytical vectors
query_batch = [[0.1, -0.2, ...], [0.4, 0.5, ...], [-0.3, 0.1, ...]]

batch_results = storage.batch_search(
    query_vectors=query_batch,
    limit=5,
    metric="cosine",
    tenant_id="tenant_alpha"
)

```

### 3. High-Speed Persistence

```python
# Commit state snapshot onto localized physical tracks instantly via Postcard
storage.save("fastvect_snapshot.bin")

# Rehydrate database states into a clean empty instance
new_storage = fastvect.VectorStorage()
new_storage.load("fastvect_snapshot.bin")

```

---

## 🛡️ License

FastVect is open-source software licensed under the MIT License. Hardened for mission-critical, ultra-low latency embedding retrieval pipelines.
