// src/bindings/storage.rs

use super::payload::parse_python_payload;
use crate::{DistanceMetric, Filter, Point, Segment};
use pyo3::prelude::*;
use pyo3::types::PyDict;

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
    ///
    /// This atomic interface orchestrates memory lock updates across spatial configurations. It converts
    /// incoming structured attributes via localized parsers and triggers automatic HNSW graph re-weaving updates.
    ///
    /// # Parameters
    /// * `point_id` - A unique unsigned 64-bit identifier linking the coordinates to its transactional registry key.
    /// * `vector` - A Python float list containing the raw coordinate tracking geometry parameters.
    /// * `payload` - An optional Python dictionary containing polymorphic data filters.
    ///
    /// # Errors
    /// Returns an operational initialization exception if underlying dictionary types fail transformation.
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

    /// Searches the high-dimensional vector space to extract top-K nearest neighbors matching target query configurations.
    ///
    /// This framework evaluates cluster density bounds at runtime to seamlessly route lookups via exact linear
    /// KNN sweeps or optimized logarithmic HNSW structural mesh traversal loops. It enforces single-stage
    /// pre-filtering if an optional tenant constraint is provided.
    ///
    /// # Parameters
    /// * `query_vector` - Target float matrix coordinates to evaluate across spatial topologies.
    /// * `limit` - The total depth matching threshold boundary (Top-K) to harvest from active memory slots.
    /// * `metric` - Configuration string matching supported parameters: `'cosine'`, `'dot_product'`, or `'euclidean'`.
    /// * `tenant_id` - An optional string key used to isolate workspace partitions under active multi-tenancy rules.
    ///
    /// # Returns
    /// A structured Python list containing tuple records formatted as: `(Point ID, Proximity Similarity Score)`.
    ///
    /// # Errors
    /// Throws a `ValueError` exception wrapper if parsing passes encounter unrecognized metric tokens.
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
                    "Unsupported metric option. Choose from: 'cosine', 'dot_product' ('dot'), or 'euclidean' ('l2').",
                ));
            }
        };

        // Construct the core filtration layer from the incoming Python runtime arguments
        let rust_filter = tenant_id.map(Filter::new);

        // Defer downstream routing paths seamlessly using our filter references
        let hits = self
            .inner
            .search(&query_vector, limit, rust_metric, rust_filter.as_ref());
        let py_results = hits.into_iter().map(|(id, score, _)| (id, score)).collect();
        Ok(py_results)
    }

    /// Commits the running transactional database state snapshot directly onto localized physical storage tracks.
    ///
    /// # Parameters
    /// * `path` - The string location defining where to construct the output binary serialization snapshot asset.
    ///
    /// # Errors
    /// Throws an asynchronous `IOError` if system partitions block file descriptor allocation workflows.
    pub fn save(&self, path: String) -> PyResult<()> {
        self.inner
            .save_to_disk(&path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e))
    }

    /// Loads and completely rehydrates a pre-existing storage binary checkpoint file back into active memory pools.
    ///
    /// # Parameters
    /// * `path` - Target system path targeting the binary database record file to extract.
    ///
    /// # Errors
    /// Throws a standard `IOError` if input buffers contain broken byte signatures or version mismatches.
    pub fn load(&self, path: String) -> PyResult<()> {
        self.inner
            .load_from_disk(&path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e))
    }
}
