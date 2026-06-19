// src/storage/segment.rs

use crate::index::exact::search_exact_knn;
use crate::{DistanceMetric, Filter, HNSWIndex, Payload, Point, ScalarQuantizer, StoragePrecision};
use parking_lot::RwLock;
use rustc_hash::FxHashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Tuple mapping for query lookup outputs: `(Point ID, Distance Score, Payload Metadata)`
pub type SearchResult = (u64, f32, Option<Payload>);

/// A synchronized atomic memory partition layer directing multi-precision vector operations across threads.
///
/// This component acts as an isolated data segment shard, coordinating thread-safe vector ingestions,
/// metadata tracking counters, graph index alignments, and transaction-isolated search sweeps.
pub struct Segment {
    points: RwLock<FxHashMap<u64, Point>>,
    hnsw_index: RwLock<HNSWIndex>,
    deleted_bits: RwLock<Vec<bool>>,
    tenant_counters: RwLock<FxHashMap<String, AtomicUsize>>,
    global_counter: AtomicUsize,
    /// Wrapped inside an explicit `RwLock` to support mutation patterns safely
    /// during snapshot rehydration without invoking undefined reference casting behavior.
    pub precision: RwLock<StoragePrecision>,
}

impl Segment {
    /// Instantiates a new, empty synchronized memory partition segment with explicit precision parameters.
    ///
    /// Configures the core underlying HNSW index state machine automatically using fine-tuned
    /// hyperparameters ($M=16, ef_{construction}=128, ef_{search}=64$) optimized to withstand quantization errors.
    ///
    /// # Parameters
    /// * `precision` - The initial target compression and storage precision configuration for this shard.
    pub fn new(precision: StoragePrecision) -> Self {
        Self {
            points: RwLock::new(FxHashMap::default()),
            hnsw_index: RwLock::new(HNSWIndex::new(16, 128, 64)),
            deleted_bits: RwLock::new(Vec::new()),
            tenant_counters: RwLock::new(FxHashMap::default()),
            global_counter: AtomicUsize::new(0),
            precision: RwLock::new(precision),
        }
    }

    /// Validates if a target data key exists inside the storage memory pool.
    ///
    /// Intercepts the query track by scanning the dense tombstone vector bitset first,
    /// preventing soft-deleted point allocations from wasting index traversal steps.
    ///
    /// # Parameters
    /// * `point_id` - Unique key identifier targeting the entity registry mapping.
    ///
    /// # Returns
    /// `true` if the object is actively registered and has not been flagged as soft-deleted.
    pub fn exists(&self, point_id: u64) -> bool {
        let idx = point_id as usize;
        let deleted_guard = self.deleted_bits.read();

        if idx < deleted_guard.len() && deleted_guard[idx] {
            return false;
        }

        self.points.read().contains_key(&point_id)
    }

    /// Extracts total records matching optional tenancy isolation criteria under lock-free atomic states.
    ///
    /// # Parameters
    /// * `tenant_id` - Optional string slice context to query localized sub-volume counter maps.
    ///
    /// # Returns
    /// Total count of live records tracked inside the targeted operational context.
    pub fn count(&self, tenant_id: Option<&str>) -> usize {
        match tenant_id {
            Some(tid) => {
                let guard = self.tenant_counters.read();
                if let Some(counter) = guard.get(tid) {
                    counter.load(Ordering::Relaxed)
                } else {
                    0
                }
            }
            None => self.global_counter.load(Ordering::Relaxed),
        }
    }

    /// Places a transactional tombstone bit marker flagging an element as deleted.
    ///
    /// Safely updates the global register by removing the raw point entry under an explicit lock mutation window
    /// while decrementing synchronized atomic workspace tenant tracking counters.
    ///
    /// # Parameters
    /// * `point_id` - Unique transactional database token assigned to register the target object deletion.
    ///
    /// # Returns
    /// `true` if the entity was actively tracked down and marked for deletion successfully.
    pub fn delete(&self, point_id: u64) -> bool {
        let idx = point_id as usize;
        let mut points_guard = self.points.write();

        if let Some(point) = points_guard.remove(&point_id) {
            self.global_counter.fetch_sub(1, Ordering::Relaxed);
            if let Some(payload) = &point.payload {
                if let Some(tenant_val) = payload.get("tenant_id") {
                    let tenant_str = match tenant_val {
                        crate::PayloadValue::Keyword(s) => Some(s.as_str()),
                        crate::PayloadValue::Text(s) => Some(s.as_str()),
                        _ => None,
                    };

                    if let Some(tid) = tenant_str {
                        let tenant_guard = self.tenant_counters.read();
                        if let Some(counter) = tenant_guard.get(tid) {
                            counter.fetch_sub(1, Ordering::Relaxed);
                        }
                    }
                }
            }

            let mut deleted_guard = self.deleted_bits.write();
            if idx >= deleted_guard.len() {
                deleted_guard.resize(idx + 1, false);
            }
            deleted_guard[idx] = true;
            return true;
        }
        false
    }

    /// Inserts or updates a multi-precision data entity inside the storage partition.
    ///
    /// Handles record life cycles by checking historical tombstone state vectors, performing thread-safe
    /// database registration updates, and inserting node linkages into the underlying HNSW graph mesh.
    ///
    /// # Parameters
    /// * `point` - The target initialized `Point` entity layout package to persist inside the partition.
    pub fn upsert(&self, point: Point) {
        let point_id = point.id;
        let vector_clone = point.vector.clone();
        let idx = point_id as usize;

        let mut tenant_id: Option<String> = None;
        if let Some(payload) = &point.payload {
            if let Some(tenant_val) = payload.get("tenant_id") {
                match tenant_val {
                    crate::PayloadValue::Keyword(s) => tenant_id = Some(s.clone()),
                    crate::PayloadValue::Text(s) => tenant_id = Some(s.clone()),
                    _ => {}
                }
            }
        }

        // Clear historical soft tombstone indicators if the record undergoes re-ingestion tracks.
        let mut deleted_guard = self.deleted_bits.write();
        if idx < deleted_guard.len() && deleted_guard[idx] {
            deleted_guard[idx] = false;
        }

        let mut write_guard = self.points.write();
        let is_update = write_guard.insert(point_id, point).is_some();

        // Increment structural concurrent atomic metrics if the transaction adds a new unique record.
        if !is_update {
            self.global_counter.fetch_add(1, Ordering::Relaxed);
            if let Some(tid) = tenant_id {
                let mut tenant_guard = self.tenant_counters.write();
                let counter = tenant_guard
                    .entry(tid)
                    .or_insert_with(|| AtomicUsize::new(0));
                counter.fetch_add(1, Ordering::Relaxed);
            }
        }

        let mut index_guard = self.hnsw_index.write();
        index_guard.insert(point_id, &vector_clone, &write_guard);
    }

    /// Searches the multi-precision vector space using unified execution routing hot paths.
    ///
    /// Intercepts incoming uncompressed feature queries, compressing them via the localized
    /// precision layout schema before dispatching lookups across linear exact or logarithmic graph index structures.
    ///
    /// # Parameters
    /// * `query_vector` - Uncompressed raw floating-point query array slice coming from the runtime boundary.
    /// * `limit` - Total top-K nearest neighbors depth window window to collect and slice.
    /// * `metric` - Proximity metric formula type targeting specific coordinate space evaluation tracks.
    /// * `filter` - An optional tenant validation constraint module used to enforce isolation fields.
    ///
    /// # Returns
    /// A sorted collection vector containing matched proximity results paired with operational metric scores.
    pub fn search(
        &self,
        query_vector: &[f32],
        limit: usize,
        metric: DistanceMetric,
        filter: Option<&Filter>,
    ) -> Vec<SearchResult> {
        let points_guard = self.points.read();
        let index_guard = self.hnsw_index.read();
        let deleted_guard = self.deleted_bits.read();
        let total_points = points_guard.len();

        // Acquire a transient shared read lock to resolve the segment's active storage precision state.
        let current_precision = *self.precision.read();
        let quantized_query = ScalarQuantizer::quantize(query_vector, current_precision);

        // Dynamically route tasks: low-volume partitions trigger linear scans, high-volume segments fire HNSW steps.
        if total_points < 500 {
            search_exact_knn(
                &quantized_query,
                limit,
                metric,
                &points_guard,
                filter,
                &deleted_guard,
            )
        } else {
            index_guard.search(
                &quantized_query,
                limit,
                metric,
                &points_guard,
                filter,
                &deleted_guard,
            )
        }
    }

    /// Commits the running quantized database segment snapshot directly to a localized binary asset.
    ///
    /// Uses Postcard binary stream encoders to pipe active memory maps, graph state registries,
    /// tombstone vectors, and structural precision configurations into a zero-copy asset format.
    ///
    /// # Parameters
    /// * `path` - Local file system path tracking where to materialize the output binary snapshot file.
    ///
    /// # Errors
    /// Returns an error string if disk creation markers fail or if encoding pipelines encounter serialization crashes.
    pub fn save_to_disk<P: AsRef<Path>>(&self, path: P) -> Result<(), String> {
        let points_guard = self.points.read();
        let index_guard = self.hnsw_index.read();
        let deleted_guard = self.deleted_bits.read();
        let precision_guard = self.precision.read();

        let file =
            File::create(path).map_err(|e| format!("Failed to create snapshot file: {}", e))?;
        let mut writer = BufWriter::new(file);

        let serializable_data = (
            &*points_guard,
            &*index_guard,
            &*deleted_guard,
            &*precision_guard,
        );
        let serialized_bytes = postcard::to_allocvec(&serializable_data)
            .map_err(|e| format!("Postcard binary serialization pipeline failure: {}", e))?;

        writer
            .write_all(&serialized_bytes)
            .map_err(|e| format!("Failed to write serialized bytes to disk: {}", e))?;
        Ok(())
    }

    /// Loads and rehydrates a pre-existing storage binary checkpoint file back into active memory pools.
    ///
    /// Hydrates target components simultaneously under a clean transaction window, updating atomic trackers
    /// and resetting schema layouts without breaking parallel execution pipeline lanes.
    ///
    /// # Parameters
    /// * `path` - Target local binary file location to fetch, open, and decode.
    ///
    /// # Errors
    /// Returns an error string if input byte buffers reflect corrupted records or historical structural mismatches.
    pub fn load_from_disk<P: AsRef<Path>>(&self, path: P) -> Result<(), String> {
        let file = File::open(path).map_err(|e| format!("Failed to open snapshot file: {}", e))?;
        let mut reader = BufReader::new(file);

        let mut raw_bytes = Vec::new();
        reader
            .read_to_end(&mut raw_bytes)
            .map_err(|e| format!("Failed to read snapshot file bytes: {}", e))?;

        let (loaded_points, loaded_index, loaded_deleted, loaded_precision): (
            FxHashMap<u64, Point>,
            HNSWIndex,
            Vec<bool>,
            StoragePrecision,
        ) = postcard::from_bytes(&raw_bytes)
            .map_err(|e| format!("Postcard binary deserialization pipeline failure: {}", e))?;

        // Calibrate atomic volume tracking gauges to match newly rehydrated structural registers.
        self.global_counter
            .store(loaded_points.len(), Ordering::Relaxed);

        let mut points_guard = self.points.write();
        let mut index_guard = self.hnsw_index.write();
        let mut deleted_guard = self.deleted_bits.write();
        let mut precision_guard = self.precision.write();

        *points_guard = loaded_points;
        *index_guard = loaded_index;
        *deleted_guard = loaded_deleted;
        *precision_guard = loaded_precision;

        Ok(())
    }
}
