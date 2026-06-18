// src/storage/segment.rs

use crate::index::exact::search_exact_knn;
use crate::{DistanceMetric, Filter, HNSWIndex, Payload, Point};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;
use std::sync::RwLock;

/// Tuple model mapping final query lookup outputs directly to structural metadata formats: (Point ID, Metrics Score, Extracted Payload)
pub type SearchResult = (u64, f32, Option<Payload>);

/// Fully-synchronized atomic memory partition layer directing high-performance operations across dynamic storage threads.
pub struct Segment {
    points: RwLock<HashMap<u64, Point>>,
    hnsw_index: RwLock<HNSWIndex>,
}

impl Segment {
    pub fn new() -> Self {
        Self {
            points: RwLock::new(HashMap::new()),
            hnsw_index: RwLock::new(HNSWIndex::new(16, 64, 32)),
        }
    }

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
    /// Seamlessly feeds the down-casted `Filter` references into routing sub-logics to enforce tenant boundary isolation rules.
    ///
    /// # Parameters
    /// * `query_vector` - A floating-point slice representing the multi-dimensional search target coordinate.
    /// * `limit` - The maximum number of nearest neighbor matches (top-K) to slice and return.
    /// * `metric` - The spatial distance formula to execute during geometric evaluation sweeps.
    /// * `filter` - An optional multi-tenancy constraint configuration wrapper.
    pub fn search(
        &self,
        query_vector: &[f32],
        limit: usize,
        metric: DistanceMetric,
        filter: Option<&Filter>,
    ) -> Vec<SearchResult> {
        let points_guard = self.points.read().unwrap();
        let index_guard = self.hnsw_index.read().unwrap();
        let total_points = points_guard.len();

        if total_points < 500 {
            search_exact_knn(query_vector, limit, metric, &points_guard, filter)
        } else {
            index_guard.search(query_vector, limit, metric, &points_guard, filter)
        }
    }

    pub fn save_to_disk<P: AsRef<Path>>(&self, path: P) -> Result<(), String> {
        let points_guard = self.points.read().unwrap();
        let index_guard = self.hnsw_index.read().unwrap();

        let file =
            File::create(path).map_err(|e| format!("Failed to create snapshot file: {}", e))?;
        let mut writer = BufWriter::new(file);

        let serializable_data = (&*points_guard, &*index_guard);

        let serialized_bytes = postcard::to_allocvec(&serializable_data)
            .map_err(|e| format!("Postcard binary serialization pipeline failure: {}", e))?;

        writer
            .write_all(&serialized_bytes)
            .map_err(|e| format!("Failed to write serialized bytes to disk: {}", e))?;

        Ok(())
    }

    pub fn load_from_disk<P: AsRef<Path>>(&self, path: P) -> Result<(), String> {
        let file = File::open(path).map_err(|e| format!("Failed to open snapshot file: {}", e))?;
        let mut reader = BufReader::new(file);

        let mut raw_bytes = Vec::new();
        reader
            .read_to_end(&mut raw_bytes)
            .map_err(|e| format!("Failed to read snapshot file bytes: {}", e))?;

        let (loaded_points, loaded_index): (HashMap<u64, Point>, HNSWIndex) =
            postcard::from_bytes(&raw_bytes).map_err(|e| {
                format!(
                    "Postcard binary deserialization pipeline failure (possible corruption): {}",
                    e
                )
            })?;

        let mut points_guard = self.points.write().unwrap();
        let mut index_guard = self.hnsw_index.write().unwrap();

        *points_guard = loaded_points;
        *index_guard = loaded_index;

        Ok(())
    }
}
