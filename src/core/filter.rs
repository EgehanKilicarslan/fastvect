// src/core/filter.rs

use serde::{Deserialize, Serialize};

/// Represents an isolated logical database query restriction used to enforce multi-tenancy constraints.
///
/// This filtering model acts as an analytical gatekeeper across vector lookup paths. By evaluating
/// structural metadata fields before compute-heavy matrix math, it successfully mitigates
/// search recall degradation drops native to post-query isolation strategies.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Filter {
    /// A unique multi-tenant identification key anchored inside shared physical storage graphs.
    pub tenant_id: String,
}

impl Filter {
    /// Instantiates a new, deterministic tenancy filtration boundary sequence.
    ///
    /// # Parameters
    /// * `tenant_id` - Target unique tenant identifier string assigned to secure spatial routing loops.
    ///
    /// # Returns
    /// An initialized `Filter` instance configured to guard workspace boundaries.
    ///
    /// # Examples
    /// ```
    /// use fastvect::core::filter::Filter;
    /// let filter = Filter::new("enterprise_tenant_alpha".to_string());
    /// ```
    pub fn new(tenant_id: String) -> Self {
        Self { tenant_id }
    }

    /// Evaluates whether a polymorphic metadata payload safely satisfies the active tenant boundary conditions.
    ///
    /// This gatekeeper scans the underlying map structures for a `"tenant_id"` lookup token. It gracefully
    /// accommodates structural deviations across polymorphic enum data layers by resolving matches
    /// seamlessly against both categorical `Keyword` tokens and standard unstructured `Text` fields.
    ///
    /// # Parameters
    /// * `payload` - An optional reference pointing to the target point entity's structured metadata map.
    ///
    /// # Returns
    /// `true` if explicit key-value parameters match active workspace isolation tokens, otherwise `false`.
    pub fn matches(&self, payload: &Option<crate::Payload>) -> bool {
        match payload {
            Some(map) => {
                if let Some(crate::PayloadValue::Keyword(tenant_val)) = map.get("tenant_id") {
                    tenant_val == &self.tenant_id
                } else if let Some(crate::PayloadValue::Text(tenant_val)) = map.get("tenant_id") {
                    tenant_val == &self.tenant_id
                } else {
                    false
                }
            }
            None => false,
        }
    }
}
