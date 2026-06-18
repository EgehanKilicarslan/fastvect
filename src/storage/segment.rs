// src/storage/segment.rs

use crate::index::exact::search_exact_knn;
use crate::{DistanceMetric, HNSWIndex, Payload, Point};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;
use std::sync::RwLock;

/// Tuple model mapping final query lookup outputs directly to structural metadata formats: (Point ID, Metrics Score, Extracted Payload)
pub type SearchResult = (u64, f32, Option<Payload>);

/// Fully-synchronized atomic memory partition layer directing high-performance operations across dynamic storage threads.
pub struct Segment {
    // Concurrent pattern isolation: RwLock permits multi-reader parallelism while constraining exclusive writer allocations
    points: RwLock<HashMap<u64, Point>>,
    hnsw_index: RwLock<HNSWIndex>,
}

impl Segment {
    /// Spawns a structurally isolated, thread-safe transactional coordinate storage memory wall segment.
    ///
    /// This constructor initializes an empty data repository backed by an analytical `HNSWIndex`
    /// configured with industry-standard hyper-parameters ($M=16$, $ef_{construction}=64$, $ef_{search}=32$).
    ///
    /// # Examples
    /// ```
    /// use fastvect::Segment;
    /// let segment = Segment::new();
    /// ```
    pub fn new() -> Self {
        Self {
            points: RwLock::new(HashMap::new()),
            hnsw_index: RwLock::new(HNSWIndex::new(16, 64, 32)), // Default analytical production presets
        }
    }

    /// Mutates or inserts a coordinates profile safely inside both the physical point matrix and the relational index mesh.
    ///
    /// This operation acquires an exclusive write lock on the underlying storage structures. It guarantees
    /// atomicity across the database state machine by concurrently injecting the vector payload into the
    /// `HashMap` data store and weaving its spatial identity directly into the active HNSW graph hierarchy.
    ///
    /// # Parameters
    /// * `point` - The foundational structural database record containing the unique ID, high-dimensional embedding vector, and optional metadata payload.
    pub fn upsert(&self, point: Point) {
        let point_id = point.id;
        let vector_clone = point.vector.clone();

        let mut write_guard = self.points.write().unwrap();
        write_guard.insert(point_id, point);

        let mut index_guard = self.hnsw_index.write().unwrap();
        index_guard.insert(point_id, &vector_clone, &write_guard);
    }

    /// Operational routing brain evaluating cluster loads to dynamically assign data searches via precise KNN execution or fast HNSW lookups.
    ///
    /// To maximize developer ergonomics and query efficiency, this orchestrator inspects the overall
    /// saturation metric of the segment. If the total payload scale is less than 500 coordinates, it routes
    /// the parameter array to a linear scan to bypass graph overheads. Otherwise, it delegates processing
    /// paths to the HNSW engine module.
    ///
    /// # Parameters
    /// * `query_vector` - A floating-point slice representing the multi-dimensional search target coordinate.
    /// * `limit` - The maximum number of nearest neighbor matches (top-K) to slice and return.
    /// * `metric` - The spatial distance formula to execute during geometric evaluation sweeps.
    ///
    /// # Returns
    /// A sorted vector collection containing matched indices paired with their mathematical proximity scores and structural metadata.
    pub fn search(
        &self,
        query_vector: &[f32],
        limit: usize,
        metric: DistanceMetric,
    ) -> Vec<SearchResult> {
        let points_guard = self.points.read().unwrap();
        let index_guard = self.hnsw_index.read().unwrap();
        let total_points = points_guard.len();

        // Execution path selection matrix: switch computational architectures seamlessly based on localized database payload scaling
        if total_points < 500 {
            search_exact_knn(query_vector, limit, metric, &points_guard)
        } else {
            index_guard.search(query_vector, limit, metric, &points_guard)
        }
    }

    /// Serializes the entire segment data (raw points and HNSW graph structures) into an optimized binary buffer using Postcard and flushes it to disk.
    ///
    /// This routine reads a point-in-time snapshot of the structural indices, serializes the aggregated mappings
    /// into a compressed memory buffer utilizing a zero-overhead binary wire layout, and commits the output sequentially
    /// into a buffered file stream to minimize operational disk overheads.
    ///
    /// # Parameters
    /// * `path` - Any target file system path reference pointing to the destination binary snapshot.
    ///
    /// # Errors
    /// Returns a descriptive error string if the platform encounters file creation failures, system I/O errors,
    /// or binary encoding pipeline drops.
    pub fn save_to_disk<P: AsRef<Path>>(&self, path: P) -> Result<(), String> {
        // Acquire read guards simultaneously to capture a consistent snapshot of the data segments
        let points_guard = self.points.read().unwrap();
        let index_guard = self.hnsw_index.read().unwrap();

        // Initialize a file stream buffered writer for optimized sequential disk writes
        let file =
            File::create(path).map_err(|e| format!("Failed to create snapshot file: {}", e))?;
        let mut writer = BufWriter::new(file);

        // Group operational states into an explicit tuple layout matrix for atomic binary serialization passes
        let serializable_data = (&*points_guard, &*index_guard);

        // Postcard serializes directly into a dynamic allocation vector buffer
        let serialized_bytes = postcard::to_allocvec(&serializable_data)
            .map_err(|e| format!("Postcard binary serialization pipeline failure: {}", e))?;

        // Flush the raw bytes directly into our buffered file stream
        writer
            .write_all(&serialized_bytes)
            .map_err(|e| format!("Failed to write serialized bytes to disk: {}", e))?;

        Ok(())
    }

    /// Reads a Postcard binary database snapshot from disk, deserializes it, and completely rehydrates the in-memory segment tracking pools.
    ///
    /// This operation reads raw binary streams sequentially into intermediate heap buffers, parses the binary layout
    /// into native maps, and acquires exclusive write locks to atomically hot-swap the active database state machine
    /// with the loaded snapshot without leaking references.
    ///
    /// # Parameters
    /// * `path` - The target path reference targeting the existing binary file to extract.
    ///
    /// # Errors
    /// Returns a descriptive error string if the snapshot file is missing, the byte streams are structurally corrupted,
    /// or if the schema encounters deserialization version-mismatch conditions.
    pub fn load_from_disk<P: AsRef<Path>>(&self, path: P) -> Result<(), String> {
        // Open file stream via a buffered reader for sequential parsing acceleration
        let file = File::open(path).map_err(|e| format!("Failed to open snapshot file: {}", e))?;
        let mut reader = BufReader::new(file);

        // Read the entire file content into a temporary heap byte buffer
        let mut raw_bytes = Vec::new();
        reader
            .read_to_end(&mut raw_bytes)
            .map_err(|e| format!("Failed to read snapshot file bytes: {}", e))?;

        // Rehydrate binary sequence tokens directly into raw target state definitions using Postcard
        let (loaded_points, loaded_index): (HashMap<u64, Point>, HNSWIndex) =
            postcard::from_bytes(&raw_bytes).map_err(|e| {
                format!(
                    "Postcard binary deserialization pipeline failure (possible corruption): {}",
                    e
                )
            })?;

        // Safely acquire write locks to clear the existing state machine and perform the memory hot-swap
        let mut points_guard = self.points.write().unwrap();
        let mut index_guard = self.hnsw_index.write().unwrap();

        *points_guard = loaded_points;
        *index_guard = loaded_index;

        Ok(())
    }
}
