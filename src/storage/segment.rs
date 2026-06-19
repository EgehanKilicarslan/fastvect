// src/storage/segment.rs

use crate::index::exact::search_exact_knn;
use crate::{DistanceMetric, Filter, HNSWIndex, Payload, Point};
use parking_lot::RwLock;
use rustc_hash::FxHashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Tuple mapping for query lookup outputs: `(Point ID, Distance Score, Payload Metadata)`
pub type SearchResult = (u64, f32, Option<Payload>);

/// A synchronized atomic memory partition layer directing vector operations across threads.
pub struct Segment {
    points: RwLock<FxHashMap<u64, Point>>,
    hnsw_index: RwLock<HNSWIndex>,
    deleted_bits: RwLock<Vec<bool>>,
    /// Thread-safe map wrapping to allow dynamic tenant registration during upserts
    tenant_counters: RwLock<FxHashMap<String, AtomicUsize>>,
    global_counter: AtomicUsize,
}

impl Segment {
    /// Instantiates a new, empty synchronized memory partition segment.
    pub fn new() -> Self {
        Self {
            points: RwLock::new(FxHashMap::default()),
            hnsw_index: RwLock::new(HNSWIndex::new(16, 64, 32)),
            deleted_bits: RwLock::new(Vec::new()),
            tenant_counters: RwLock::new(FxHashMap::default()),
            global_counter: AtomicUsize::new(0),
        }
    }

    /// Validates if a target data key exists inside the storage memory pool.
    pub fn exists(&self, point_id: u64) -> bool {
        let idx = point_id as usize;
        let deleted_guard = self.deleted_bits.read();

        if idx < deleted_guard.len() && deleted_guard[idx] {
            return false;
        }

        self.points.read().contains_key(&point_id)
    }

    /// Extracts total records matching optional tenancy isolation criteria.
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

    /// Inserts or updates a high-dimensional vector coordinate paired with structured payloads.
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

        let mut deleted_guard = self.deleted_bits.write();
        if idx < deleted_guard.len() && deleted_guard[idx] {
            deleted_guard[idx] = false;
        }

        let mut write_guard = self.points.write();
        let is_update = write_guard.insert(point_id, point).is_some();

        if !is_update {
            self.global_counter.fetch_add(1, Ordering::Relaxed);
            if let Some(tid) = tenant_id {
                let mut tenant_guard = self.tenant_counters.write();
                // Dynamic initialization: create the atomic counter if it doesn't exist yet
                let counter = tenant_guard
                    .entry(tid)
                    .or_insert_with(|| AtomicUsize::new(0));
                counter.fetch_add(1, Ordering::Relaxed);
            }
        }

        let mut index_guard = self.hnsw_index.write();
        index_guard.insert(point_id, &vector_clone, &write_guard);
    }

    /// Searches the high-dimensional vector space using dynamic runtime execution routing paths.
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

        if total_points < 500 {
            search_exact_knn(
                query_vector,
                limit,
                metric,
                &points_guard,
                filter,
                &deleted_guard,
            )
        } else {
            index_guard.search(
                query_vector,
                limit,
                metric,
                &points_guard,
                filter,
                &deleted_guard,
            )
        }
    }

    /// Commits the running in-memory database segment snapshot directly to a localized binary asset.
    pub fn save_to_disk<P: AsRef<Path>>(&self, path: P) -> Result<(), String> {
        let points_guard = self.points.read();
        let index_guard = self.hnsw_index.read();
        let deleted_guard = self.deleted_bits.read();

        let file =
            File::create(path).map_err(|e| format!("Failed to create snapshot file: {}", e))?;
        let mut writer = BufWriter::new(file);

        let serializable_data = (&*points_guard, &*index_guard, &*deleted_guard);
        let serialized_bytes = postcard::to_allocvec(&serializable_data)
            .map_err(|e| format!("Postcard binary serialization pipeline failure: {}", e))?;

        writer
            .write_all(&serialized_bytes)
            .map_err(|e| format!("Failed to write serialized bytes to disk: {}", e))?;
        Ok(())
    }

    /// Loads and rehydrates a pre-existing storage binary checkpoint file back into active memory pools.
    pub fn load_from_disk<P: AsRef<Path>>(&self, path: P) -> Result<(), String> {
        let file = File::open(path).map_err(|e| format!("Failed to open snapshot file: {}", e))?;
        let mut reader = BufReader::new(file);

        let mut raw_bytes = Vec::new();
        reader
            .read_to_end(&mut raw_bytes)
            .map_err(|e| format!("Failed to read snapshot file bytes: {}", e))?;

        let (loaded_points, loaded_index, loaded_deleted): (
            FxHashMap<u64, Point>,
            HNSWIndex,
            Vec<bool>,
        ) = postcard::from_bytes(&raw_bytes)
            .map_err(|e| format!("Postcard binary deserialization pipeline failure: {}", e))?;

        self.global_counter
            .store(loaded_points.len(), Ordering::Relaxed);

        let mut points_guard = self.points.write();
        let mut index_guard = self.hnsw_index.write();
        let mut deleted_guard = self.deleted_bits.write();

        *points_guard = loaded_points;
        *index_guard = loaded_index;
        *deleted_guard = loaded_deleted;

        Ok(())
    }
}
