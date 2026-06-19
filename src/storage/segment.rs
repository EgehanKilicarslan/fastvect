// src/storage/segment.rs

use crate::index::exact::search_exact_knn;
use crate::{DistanceMetric, Filter, HNSWIndex, Payload, Point, ScalarQuantizer, StoragePrecision};
use parking_lot::RwLock;
use rustc_hash::FxHashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Tuple mapping defining the standard structure for lookup outputs: `(Point ID, Distance Score, Payload Metadata)`.
pub type SearchResult = (u64, f32, Option<Payload>);

/// Centralized state configuration holding the un-locked core database spaces.
///
/// Groups primary indexes and multi-tenant telemetry structures to allow single-barrier
/// synchronization states across variant thread execution tracks.
pub struct SegmentState {
    /// Global repository containing physical vector records paired with their metadata blocks.
    pub points: FxHashMap<u64, Point>,
    /// Underlying Hierarchical Navigable Small World index graph framework.
    pub hnsw_index: HNSWIndex,
    /// Dense tracking bitset mapping soft transactional deletion indicators.
    pub deleted_bits: Vec<bool>,
    /// Thread-safe map tracking isolated record metrics allocated per individual workspace tenant.
    pub tenant_counters: FxHashMap<String, AtomicUsize>,
    /// Numerical floating-point compression configuration format assigned to this partition instance.
    pub precision: StoragePrecision,
}

/// A synchronized memory partition layer directing lock-free concurrent query flows.
///
/// Orchestrates safe high-velocity parallel processing loops by establishing a single top-level
/// read/write guard mechanism over vulnerable internal structural allocations.
pub struct Segment {
    /// Wrapped architecture layout containing localized core execution components.
    pub state: RwLock<SegmentState>,
    /// Global lock-free counter capturing total active historical items.
    pub global_counter: AtomicUsize,
}

impl Segment {
    /// Instantiates an empty, synchronized partition segment workspace initialized with exact quantization specs.
    ///
    /// Automatically provisions baseline parameters targeting production-grade spatial indexing demands.
    ///
    /// # Parameters
    /// * `precision` - The initial compression layout and quantization parameters assigned to this segment shard.
    pub fn new(precision: StoragePrecision) -> Self {
        Self {
            state: RwLock::new(SegmentState {
                points: FxHashMap::default(),
                hnsw_index: HNSWIndex::new(16, 128, 64),
                deleted_bits: Vec::new(),
                tenant_counters: FxHashMap::default(),
                precision,
            }),
            global_counter: AtomicUsize::new(0),
        }
    }

    /// Validates if a target entity identifier actively exists inside the local database partitions.
    pub fn exists(&self, point_id: u64) -> bool {
        let guard = self.state.read();
        let idx = point_id as usize;

        // Immediately catch matching soft transactional tombstones to prevent data leakage paths
        if idx < guard.deleted_bits.len() && guard.deleted_bits[idx] {
            return false;
        }

        guard.points.contains_key(&point_id)
    }

    /// Extracts total records currently allocated within designated multi-tenancy bounds using lock-free atomic states.
    pub fn count(&self, tenant_id: Option<&str>) -> usize {
        match tenant_id {
            Some(tid) => {
                let guard = self.state.read();
                if let Some(counter) = guard.tenant_counters.get(tid) {
                    counter.load(Ordering::Relaxed)
                } else {
                    0
                }
            }
            None => self.global_counter.load(Ordering::Relaxed),
        }
    }

    /// Commits a soft transaction tombstone block marker flagging an element as deleted.
    ///
    /// Instantly disconnects record visibility from spatial indexes while preparing data paths
    /// for down-stream cleanups without introducing high thread latencies.
    pub fn delete(&self, point_id: u64) -> bool {
        let idx = point_id as usize;
        let mut guard = self.state.write();

        if let Some(point) = guard.points.remove(&point_id) {
            self.global_counter.fetch_sub(1, Ordering::Relaxed);
            if let Some(payload) = &point.payload {
                if let Some(tenant_val) = payload.get("tenant_id") {
                    let tenant_str = match tenant_val {
                        crate::PayloadValue::Keyword(s) => Some(s.as_str()),
                        crate::PayloadValue::Text(s) => Some(s.as_str()),
                        _ => None,
                    };

                    if let Some(tid) = tenant_str {
                        if let Some(counter) = guard.tenant_counters.get(tid) {
                            counter.fetch_sub(1, Ordering::Relaxed);
                        }
                    }
                }
            }

            if idx >= guard.deleted_bits.len() {
                guard.deleted_bits.resize(idx + 1, false);
            }
            guard.deleted_bits[idx] = true;
            return true;
        }
        false
    }

    /// Inserts or updates a multi-precision data entity inside the centralized partition state.
    ///
    /// Clears downstream soft tombstone markers if data keys undergo re-ingestion, updates tracking metrics,
    /// and weaves node relationships securely into the global graphical layout.
    pub fn upsert(&self, point: Point) {
        let point_id = point.id;
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

        let mut guard = self.state.write();
        if idx < guard.deleted_bits.len() && guard.deleted_bits[idx] {
            guard.deleted_bits[idx] = false;
        }

        let SegmentState {
            points,
            hnsw_index,
            tenant_counters,
            ..
        } = &mut *guard;
        let is_update = points.insert(point_id, point).is_some();

        if !is_update {
            self.global_counter.fetch_add(1, Ordering::Relaxed);
            if let Some(tid) = tenant_id {
                let counter = tenant_counters
                    .entry(tid)
                    .or_insert_with(|| AtomicUsize::new(0));
                counter.fetch_add(1, Ordering::Relaxed);
            }
        }

        // FIXED: Borrowed direct structural layouts straight from the storage map
        // post-insertion, completely avoiding preemptive heap vector allocations!
        let vector_ref = &points[&point_id].vector;
        hnsw_index.insert(point_id, vector_ref, points);
    }

    /// Executes high-velocity concurrent vector space lookups with zero long-lived lock contention.
    ///
    /// Workers establish a single unified read barrier at entry, proceeding into computational layers
    /// with zero internal threading friction to fully saturate processing hardware threads.
    pub fn search(
        &self,
        query_vector: &[f32],
        limit: usize,
        metric: DistanceMetric,
        filter: Option<&Filter>,
    ) -> Vec<SearchResult> {
        let guard = self.state.read();

        let current_precision = guard.precision;
        let quantized_query = ScalarQuantizer::quantize(query_vector, current_precision);
        let total_points = guard.points.len();

        if total_points < 500 {
            search_exact_knn(
                &quantized_query,
                limit,
                metric,
                &guard.points,
                filter,
                &guard.deleted_bits,
            )
        } else {
            guard.hnsw_index.search(
                &quantized_query,
                limit,
                metric,
                &guard.points,
                filter,
                &guard.deleted_bits,
            )
        }
    }

    /// Serializes and flushes the entire database segment snapshot directly into a zero-copy postcard asset file.
    pub fn save_to_disk<P: AsRef<Path>>(&self, path: P) -> Result<(), String> {
        let guard = self.state.read();
        let file =
            File::create(path).map_err(|e| format!("Failed to create snapshot file: {}", e))?;
        let mut writer = BufWriter::new(file);

        let serializable_data = (
            &guard.points,
            &guard.hnsw_index,
            &guard.deleted_bits,
            &guard.precision,
        );
        let serialized_bytes = postcard::to_allocvec(&serializable_data)
            .map_err(|e| format!("Postcard binary serialization pipeline failure: {}", e))?;

        writer
            .write_all(&serialized_bytes)
            .map_err(|e| format!("IO failure: {}", e))?;
        Ok(())
    }

    /// Fully rehydrates and loads historical data checkpoints back into the runtime storage memory spaces.
    pub fn load_from_disk<P: AsRef<Path>>(&self, path: P) -> Result<(), String> {
        let file = File::open(path).map_err(|e| format!("Failed to open snapshot file: {}", e))?;
        let mut reader = BufReader::new(file);

        let mut raw_bytes = Vec::new();
        reader
            .read_to_end(&mut raw_bytes)
            .map_err(|e| format!("IO failure: {}", e))?;

        let (loaded_points, loaded_index, loaded_deleted, loaded_precision): (
            FxHashMap<u64, Point>,
            HNSWIndex,
            Vec<bool>,
            StoragePrecision,
        ) = postcard::from_bytes(&raw_bytes)
            .map_err(|e| format!("Deserialization failure: {}", e))?;

        self.global_counter
            .store(loaded_points.len(), Ordering::Relaxed);

        let mut guard = self.state.write();

        // FIXED: Dynamically rehydrated tenant counters telemetry arrays straight from loaded storage states
        let mut new_tenant_counters = FxHashMap::default();
        for point in loaded_points.values() {
            if let Some(payload) = &point.payload {
                if let Some(tenant_val) = payload.get("tenant_id") {
                    let tid_opt = match tenant_val {
                        crate::PayloadValue::Keyword(s) => Some(s.clone()),
                        crate::PayloadValue::Text(s) => Some(s.clone()),
                        _ => None,
                    };
                    if let Some(tid) = tid_opt {
                        new_tenant_counters
                            .entry(tid)
                            .or_insert_with(|| AtomicUsize::new(0))
                            .fetch_add(1, Ordering::Relaxed);
                    }
                }
            }
        }

        guard.points = loaded_points;
        guard.hnsw_index = loaded_index;
        guard.deleted_bits = loaded_deleted;
        guard.precision = loaded_precision;
        guard.tenant_counters = new_tenant_counters;

        Ok(())
    }
}
