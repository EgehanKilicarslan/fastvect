# fastvect.pyi
from typing import Literal

# Supported distance metric literals mapped precisely across FFI bridges
SupportedMetrics = Literal["cosine", "dot_product", "euclidean", "dot", "l2"]

class VectorStorage:
    """
    Qdrant-inspired high-performance embedded Vector Storage and Search engine.

    Powered by a robust Rust core, managing synchronized transactional memory states,
    opportunistic macro/micro HNSW graph index routing, and zero-overhead Postcard serialization.
    """

    def __init__(self) -> None:
        """
        Instantiates an isolated, production-grade VectorStorage workspace environment.
        Allocates thread-safe data partitions and pre-configures HNSW hyper-parameters.
        """
        ...

    def upsert(
        self,
        point_id: int,
        vector: list[float],
        payload: dict[str, str | int | float] | None = None,
    ) -> None:
        """
        Universally inserts or updates a coordinate entity embedding paired with structured metadata.

        This operation triggers atomic mutations across internal storage pools and schedules
        immediate bi-directional link weaving within the active HNSW graph hierarchies.

        Args:
            point_id: Unique 64-bit unsigned tracking key.
            vector: High-dimensional raw floating-point coordinate array.
            payload: Optional unstructured dictionary mapping string keys to polymorphic primitives.
        """
        ...

    def search(
        self,
        query_vector: list[float],
        limit: int,
        metric: SupportedMetrics,
        tenant_id: str | None = None,
    ) -> list[tuple[int, float]]:
        """
        Searches the high-dimensional vector space to extract the Top-K nearest neighbors.

        Dynamically routes queries via exact linear $O(N)$ brute-force sweeps or ultra-fast
        logarithmic $O(\\log N)$ HNSW hierarchical graph traversals depending on data volume thresholds.
        Enforces single-stage metadata pre-filtering if tenancy constraints are assigned.

        Args:
            query_vector: Analytical float coordinates used as the lookup search target.
            limit: Total result count capacity boundary depth (Top-K matching limits).
            metric: Proximity formula token. Supported configurations: 'cosine', 'dot_product', 'euclidean'.
            tenant_id: Optional identification tag string used to enforce secure workspace isolation.

        Returns:
            A ordered list of records matching the schema layout: `[(Point ID, Metric Match Score), ...]`
        """
        ...

    def save(self, path: str) -> None:
        """
        Commits the active in-memory database segment snapshot directly to a localized binary asset.

        Utilizes a highly compressed, zero-copy serialization scheme via Postcard with sequential
        buffered disk writer pipelines.

        Args:
            path: Target local file system path where the output checkpoint asset should be written.

        Raises:
            IOError: If platform storage access rights block file descriptor synchronization workflows.
        """
        ...

    def load(self, path: str) -> None:
        """
        Loads and completely rehydrates a pre-existing storage binary checkpoint file back into memory.

        Acquires exclusive process locks to clear current states and perform a zero-leak memory hot-swap.

        Args:
            path: Target local binary backup snapshot location to fetch and parse.

        Raises:
            IOError: If input byte buffers are physically corrupted or reflect historical schema mismatches.
        """
        ...
