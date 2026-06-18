// src/bindings/payload.rs

use crate::{Payload, PayloadValue};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyFloat, PyInt, PyString};
use std::collections::HashMap;

/// Safely parses a Python dictionary into a native Rust polymorphic `Payload` structure.
///
/// This bridge routine inspects fluid, dynamically-typed Python dictionaries using deterministic
/// runtime type-matching (`is_instance_of`). It maps primitive Python values directly onto
/// strongly-typed Rust internal enum variants, protecting memory boundaries across the FFI wall.
///
/// # Heuristics
/// * Short metadata string types ($\le 64$ characters) are automatically routed into the `Keyword` token variant.
/// * High-capacity string sequences ($> 64$ characters) are seamlessly cataloged as full analytical `Text` fields.
///
/// # Parameters
/// * `dict` - A reference pointer targeting a Python dictionary (`PyDict`) bound to the current GIL lifetime scope.
///
/// # Errors
/// Returns a `PyResult::Err` containing a virtual `TypeError` if extraction tokens encounter corrupted memory formatting.
pub fn parse_python_payload(dict: &Bound<'_, PyDict>) -> PyResult<Payload> {
    let mut map = HashMap::new();

    for (key, value) in dict.iter() {
        // Extract the map string key safely across runtime boundaries
        let k_str = key.extract::<String>()?;

        // Explicit type matching matching Qdrant payload flexibility
        if value.is_instance_of::<PyString>() {
            let p_str = value.extract::<String>()?;

            // Boundary length heuristic separating categorizations from unstructured text models
            if p_str.len() <= 64 {
                map.insert(k_str, PayloadValue::Keyword(p_str));
            } else {
                map.insert(k_str, PayloadValue::Text(p_str));
            }
        } else if value.is_instance_of::<PyInt>() {
            let p_int = value.extract::<i64>()?;
            map.insert(k_str, PayloadValue::Integer(p_int));
        } else if value.is_instance_of::<PyFloat>() {
            let p_float = value.extract::<f64>()?;
            map.insert(k_str, PayloadValue::Float(p_float));
        }
    }

    Ok(map)
}
