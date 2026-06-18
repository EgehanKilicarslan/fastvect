// src/core/vector.rs

use std::collections::HashMap;

/// Represents the polymorphic primitive variants permitted inside the schema-less unstructured metadata payload dictionary.
#[derive(Clone, Debug, PartialEq)]
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
pub type Payload = HashMap<String, PayloadValue>;

/// The foundational atomic architectural entity tracked within the vector index cluster.
#[derive(Clone, Debug)]
pub struct Point {
    /// Unique transactional identifier mapped explicitly to the embedding entity registry.
    pub id: u64,
    /// High-dimensional floating-point embedding arrays capturing granular mathematical relationships.
    pub vector: Vec<f32>,
    /// Highly optional schemaless metadata storage assigned for pre/post structural search filtering matrices.
    pub payload: Option<Payload>,
}
