# fastvect.pyi
from typing import Literal

SupportedMetrics = Literal["cosine", "dot_product", "euclidean", "dot", "l2"]

class VectorStorage:
    """
    High-performance embedded Vector Storage and Search engine.

    Powered by an optimized Rust core managing synchronized memory states,
    HNSW graph index routing pipelines, and compressed binary state serialization.
    """

    def __init__(self) -> None:
        """
        Instantiates an empty synchronized VectorStorage workspace partition.
        """
        ...

    def upsert(
        self,
        point_id: int,
        vector: list[float],
        payload: dict[str, str | int | float] | None = None,
    ) -> None:
        """
        Inserts or updates a high-dimensional vector entity paired with payload metadata.

        Clears historical tombstone markers matching the identifier if the key undergoes
        re-insertion tracks.

        Args:
            point_id: Unique unsigned 64-bit entity key identifier.
            vector: High-dimensional raw floating-point coordinate array.
            payload: Optional dictionary mapping string keys to polymorphic primitive values.
        """
        ...

    def exists(self, point_id: int) -> bool:
        """
        Validates if a target data key exists inside the storage memory pool.

        Args:
            point_id: Unique unsigned 64-bit entity key identifier.

        Returns:
            True if the entity is registered and has not been soft-deleted via tombstones.
        """
        ...

    def count(self, tenant_id: str | None = None) -> int:
        """
        Extracts total records active within specified boundary contexts under lock-free states.

        Args:
            tenant_id: Optional string key targeting a specific isolated workspace tenant.

        Returns:
            Total count of live records active within the partition pool.
        """
        ...

    def delete(self, point_id: int) -> bool:
        """
        Places a transactional tombstone bit marker flagging an element as deleted.

        Executes a soft-delete mutation by updating internal tracking states without
        corrupting active HNSW graph connection topologies.

        Args:
            point_id: Unique unsigned 64-bit entity key identifier to mark for deletion.

        Returns:
            True if the object was successfully found and marked for deletion.
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
        Searches the high-dimensional vector space using dynamic execution routing paths.

        Evaluates volume thresholds to route lookups through precise linear KNN sweeps
        or ultra-fast logarithmic HNSW hierarchical graph traversals.

        Args:
            query_vector: High-dimensional source list coordinate array used as the lookup target.
            limit: Total capacity depth matching threshold boundaries (Top-K results).
            metric: Spatial distance metric formula used to compute similarity scores.
            tenant_id: Optional string token used to restrict queries to isolated workspaces.

        Returns:
            An ordered list mapping proximity matches: `[(Point ID, Similarity Score), ...]`
        """
        ...

    def batch_search(
        self,
        query_vectors: list[list[float]],
        limit: int,
        metric: SupportedMetrics,
        tenant_id: str | None = None,
    ) -> list[list[tuple[int, float]]]:
        """
        Executes concurrent high-dimensional batch vector lookups via multi-threaded maps.

        Bypasses Python runtime loop overheads and GIL bottlenecks by driving multi-tenant graph
        filtering routines simultaneously across available hardware processing threads.

        Args:
            query_vectors: A nested list containing multiple query vector arrays to evaluate.
            limit: Total capacity depth matching threshold boundaries per query (Top-K results).
            metric: Spatial distance metric formula used to compute similarity scores.
            tenant_id: Optional string token used to restrict queries to isolated workspaces.

        Returns:
            A nested list containing ordered matching records arrays: `[[(Point ID, Score), ...], ...]`
        """
        ...

    def save(self, path: str) -> None:
        """
        Commits running in-memory database segment snapshots directly to a localized binary asset.

        Args:
            path: Target local file system path where the checkpoint asset should be written.

        Raises:
            IOError: If platform storage access rights block file descriptor synchronization workflows.
        """
        ...

    def load(self, path: str) -> None:
        """
        Loads and rehydrates a pre-existing storage binary checkpoint file back into memory.

        Args:
            path: Target local binary backup snapshot location to fetch and parse.

        Raises:
            IOError: If input byte buffers are physically corrupted or reflect historical schema mismatches.
        """
        ...
