// src/core/vector.rs

use crate::{QuantizedVector, ScalarQuantizer, StoragePrecision};
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};

/// Represents the polymorphic primitive variants permitted inside the schema-less unstructured metadata payload dictionary.
///
/// This enum enables heterogeneous metadata tracking attached to spatial points, allowing the database
/// orchestration layer to execute structured pre-filtering or post-filtering pipelines during proximity sweeps.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum PayloadValue {
    /// Optimized for exact matches, indexing tokens, relational tags, or low-cardinality flags (e.g., "production", "user_901").
    Keyword(String),
    /// Tailored for high-capacity raw string chunks, natural language text, or serialized JSON responses (e.g., semantic caching layers).
    Text(String),
    /// Designated for relational database primary keys, incremental UNIX timestamps, or mathematical counters.
    Integer(i64),
    /// Reserved for positional scores, proximity thresholds, or localized geographical coordinate offsets.
    Float(f64),
}

/// A structured, concurrent-safe dictionary hashmap binding string identifiers to polymorphic metadata values.
///
/// It acts as the secondary storage schema mapping attributes to dynamic indices for conditional logical filtering.
pub type Payload = FxHashMap<String, PayloadValue>;

/// The foundational atomic architectural entity tracked within the vector index cluster.
///
/// A `Point` encapsulates the operational identity, multi-precision compressed embedding layout,
/// and peripheral unstructured dynamic attributes that construct the underlying data mesh topology.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Point {
    /// Unique transactional identifier mapped explicitly to the embedding entity registry.
    pub id: u64,
    /// Encapsulated multi-precision quantized vector data partition workspace.
    pub vector: QuantizedVector,
    /// Highly optional schemaless metadata storage assigned for pre/post structural search filtering matrices.
    pub payload: Option<Payload>,
}

impl Point {
    /// Factory initializer providing transparent inline quantization conversions for raw coordinate blocks.
    ///
    /// Intercepts the original uncompressed floating-point coordinates at the ingestion boundary,
    /// routing them directly into compression pipelines before embedding them into the structured `Point` wrapper.
    ///
    /// # Parameters
    /// * `id` - Unique tracking key mapped explicitly to the target entity registry.
    /// * `raw_vec` - Uncompressed original coordinate array slice.
    /// * `precision` - Targeted memory layout and compression variant configuration (F32, F16, or F8).
    /// * `payload` - Optional dictionary mapping string keys to polymorphic primitive filtering values.
    ///
    /// # Returns
    /// An initialized `Point` entity fully prepared for graph insertion and storage layer registration tracks.
    pub fn new_quantized(
        id: u64,
        raw_vec: &[f32],
        precision: StoragePrecision,
        payload: Option<Payload>,
    ) -> Self {
        Self {
            id,
            vector: ScalarQuantizer::quantize(raw_vec, precision),
            payload,
        }
    }
}
