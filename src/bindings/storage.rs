// src/bindings/storage.rs

use super::payload::parse_python_payload;
use crate::{DistanceMetric, Filter, Point, Segment};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;

/// Qdrant-inspired high-performance Vector Storage and Search engine exposed directly to Python runtimes.
///
/// This struct functions as a safe, concurrent architectural encapsulation wrapper managing an isolated
/// memory partition segment layer. It supports lock-free multi-threaded query streams and sequential persistence pipelines.
#[pyclass(name = "VectorStorage")]
pub struct PyVectorStorage {
    inner: Segment,
}

#[pymethods]
impl PyVectorStorage {
    /// Instantiates a new, isolated production-grade `VectorStorage` workspace environment.
    #[new]
    pub fn new() -> Self {
        Self {
            inner: Segment::new(),
        }
    }

    /// Universally inserts or updates a coordinate entity embedding paired with dynamic structured metadata payloads.
    pub fn upsert(
        &self,
        point_id: u64,
        vector: Vec<f32>,
        payload: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<()> {
        let rust_payload = match payload {
            Some(dict) => Some(parse_python_payload(dict)?),
            None => None,
        };

        let point = Point {
            id: point_id,
            vector,
            payload: rust_payload,
        };

        self.inner.upsert(point);
        Ok(())
    }

    /// Verifies if a target identifier exists inside the active memory pool partitions.
    ///
    /// # Parameters
    /// * `point_id` - Unique tracking key assigned to register the target object.
    ///
    /// # Returns
    /// `true` if the entity is registered and has not been flagged by a soft tombstone erasure.
    pub fn exists(&self, point_id: u64) -> bool {
        self.inner.exists(point_id)
    }

    /// Extracts total records currently tracked inside active partition pools under lock-free constraints.
    ///
    /// # Parameters
    /// * `tenant_id` - Optional string key targeting a specific isolated workspace tenant environment.
    ///
    /// # Returns
    /// Total count of live records matching specified spatial context parameters.
    #[pyo3(signature = (tenant_id=None))]
    pub fn count(&self, tenant_id: Option<String>) -> usize {
        self.inner.count(tenant_id.as_deref())
    }

    /// Commits a soft transaction block marker flagging an element as deleted.
    ///
    /// # Parameters
    /// * `point_id` - Unique transactional database token assigned to register the target object deletion.
    ///
    /// # Returns
    /// `true` if the entity was tracked down and marked for deletion successfully.
    pub fn delete(&self, point_id: u64) -> bool {
        self.inner.delete(point_id)
    }

    /// Searches the high-dimensional vector space to extract top-K nearest neighbors matching target query configurations.
    #[pyo3(signature = (query_vector, limit, metric, tenant_id=None))]
    pub fn search(
        &self,
        query_vector: Vec<f32>,
        limit: usize,
        metric: String,
        tenant_id: Option<String>,
    ) -> PyResult<Vec<(u64, f32)>> {
        let rust_metric = match metric.to_lowercase().as_str() {
            "dot_product" | "dot" => DistanceMetric::DotProduct,
            "cosine" => DistanceMetric::Cosine,
            "euclidean" | "l2" => DistanceMetric::HighPrecisionEuclidean,
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Unsupported metric option. Choose from: 'cosine', 'dot_product', or 'euclidean'.",
                ));
            }
        };

        let rust_filter = tenant_id.map(Filter::new);
        let hits = self
            .inner
            .search(&query_vector, limit, rust_metric, rust_filter.as_ref());
        let py_results = hits.into_iter().map(|(id, score, _)| (id, score)).collect();
        Ok(py_results)
    }

    /// Executes concurrent high-dimensional batch vector lookups across available hardware processing units.
    #[pyo3(signature = (query_vectors, limit, metric, tenant_id=None))]
    pub fn batch_search(
        &self,
        query_vectors: Vec<Vec<f32>>,
        limit: usize,
        metric: String,
        tenant_id: Option<String>,
    ) -> PyResult<Vec<Vec<(u64, f32)>>> {
        let rust_metric = match metric.to_lowercase().as_str() {
            "dot_product" | "dot" => DistanceMetric::DotProduct,
            "cosine" => DistanceMetric::Cosine,
            "euclidean" | "l2" => DistanceMetric::HighPrecisionEuclidean,
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Unsupported metric option assigned. Choose from: 'cosine', 'dot_product', or 'euclidean'.",
                ));
            }
        };

        let rust_filter = tenant_id.map(Filter::new);
        let batch_results: Vec<Vec<(u64, f32)>> = query_vectors
            .par_iter()
            .map(|query_vector| {
                let hits =
                    self.inner
                        .search(query_vector, limit, rust_metric, rust_filter.as_ref());
                hits.into_iter().map(|(id, score, _)| (id, score)).collect()
            })
            .collect();

        Ok(batch_results)
    }

    /// Commits the running transactional database state snapshot directly onto localized physical storage tracks.
    pub fn save(&self, path: String) -> PyResult<()> {
        self.inner
            .save_to_disk(&path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e))
    }

    /// Loads and completely rehydrates a pre-existing storage binary checkpoint file back into active memory pools.
    pub fn load(&self, path: String) -> PyResult<()> {
        self.inner
            .load_from_disk(&path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e))
    }
}
