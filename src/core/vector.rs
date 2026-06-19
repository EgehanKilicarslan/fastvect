// src/core/vector.rs

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
/// A `Point` encapsulates the operational identity, high-dimensional relational characteristics (embeddings),
/// and peripheral unstructured dynamic attributes that construct the underlying data mesh topology.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Point {
    /// Unique transactional identifier mapped explicitly to the embedding entity registry.
    pub id: u64,
    /// High-dimensional floating-point embedding arrays capturing granular mathematical relationships (e.g., 1536-dimensional arrays).
    pub vector: Vec<f32>,
    /// Highly optional schemaless metadata storage assigned for pre/post structural search filtering matrices.
    pub payload: Option<Payload>,
}
