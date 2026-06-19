// src/bindings/storage.rs

use super::payload::parse_python_payload;
use crate::{DistanceMetric, Filter, Point, Segment, StoragePrecision};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;

#[pyclass(name = "VectorStorage")]
pub struct PyVectorStorage {
    inner: Segment,
}

#[pymethods]
impl PyVectorStorage {
    #[new]
    #[pyo3(signature = (precision = "f32".to_string()))]
    pub fn new(precision: String) -> PyResult<Self> {
        let rust_precision = match precision.to_lowercase().as_str() {
            "f32" => StoragePrecision::F32,
            "f16" => StoragePrecision::F16,
            "f8" | "int8" => StoragePrecision::F8,
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Unsupported precision option assigned. Choose from: 'f32', 'f16', or 'f8'.",
                ));
            }
        };

        Ok(Self {
            inner: Segment::new(rust_precision),
        })
    }

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

        // FIXED: Resolved target field matching to extract precision through the inner segment architecture field
        let active_precision = self.inner.state.read().precision;

        let point = Point::new_quantized(point_id, &vector, active_precision, rust_payload);

        self.inner.upsert(point);
        Ok(())
    }

    pub fn exists(&self, point_id: u64) -> bool {
        self.inner.exists(point_id)
    }

    #[pyo3(signature = (tenant_id=None))]
    pub fn count(&self, tenant_id: Option<String>) -> usize {
        self.inner.count(tenant_id.as_deref())
    }

    pub fn delete(&self, point_id: u64) -> bool {
        self.inner.delete(point_id)
    }

    #[pyo3(signature = (query_vector, limit, metric, tenant_id=None))]
    pub fn search(
        &self,
        py: Python<'_>,
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

        // DOCUMENTATION STANDARD: Temporarily drops the GIL to let other threads process concurrent loops
        let hits = py.detach(|| {
            self.inner
                .search(&query_vector, limit, rust_metric, rust_filter.as_ref())
        });

        let py_results = hits.into_iter().map(|(id, score, _)| (id, score)).collect();
        Ok(py_results)
    }

    #[pyo3(signature = (query_vectors, limit, metric, tenant_id=None))]
    pub fn batch_search(
        &self,
        py: Python<'_>,
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

        // DOCUMENTATION STANDARD: Leveraged py.detach to safely unblock Rayon worker threads
        let batch_results: Vec<Vec<(u64, f32)>> = py.detach(|| {
            query_vectors
                .par_iter()
                .map(|query_vector| {
                    let hits =
                        self.inner
                            .search(query_vector, limit, rust_metric, rust_filter.as_ref());
                    hits.into_iter().map(|(id, score, _)| (id, score)).collect()
                })
                .collect()
        });

        Ok(batch_results)
    }

    pub fn save(&self, path: String) -> PyResult<()> {
        self.inner
            .save_to_disk(&path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e))
    }

    pub fn load(&self, path: String) -> PyResult<()> {
        self.inner
            .load_from_disk(&path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e))
    }
}
