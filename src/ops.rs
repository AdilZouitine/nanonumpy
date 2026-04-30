//! Shared operation types, validation, and allocation.
//!
//! Python developers can read this module as the "operation layer". `lib.rs`
//! converts Python inputs into Rust data, then this module checks shape
//! invariants and chooses either the naive Rust loop or the SIMD dispatcher.

use crate::dispatch;
use crate::naive_rust;

#[derive(Clone, Copy, Debug)]
pub enum Op {
    /// Elementwise `a[i] + b[i]`.
    Add,
    /// Elementwise `a[i] - b[i]`.
    Sub,
    /// Elementwise `a[i] * b[i]`.
    Mul,
    /// Elementwise `a[i] / b[i]`.
    Div,
}

impl Op {
    /// Apply this operation to one scalar pair.
    ///
    /// SIMD code uses vector intrinsics for full chunks, but the scalar
    /// fallback and tail handling both use this exact operation.
    pub fn apply(self, x: f32, y: f32) -> f32 {
        match self {
            Self::Add => x + y,
            Self::Sub => x - y,
            Self::Mul => x * y,
            Self::Div => x / y,
        }
    }
}

/// Validate that both inputs describe the same one-dimensional length.
///
/// The project intentionally does not implement broadcasting. A mismatch is
/// therefore a user error, surfaced to Python as `ValueError`.
pub fn validate_lengths(a: &[f32], b: &[f32]) -> Result<(), String> {
    if a.len() != b.len() {
        return Err("arrays must have the same length".to_string());
    }

    Ok(())
}

/// Allocate a Python-list-style result and fill it with the naive Rust loop.
///
/// This is the implementation behind `add_rust`, `sub_rust`, `mul_rust`, and
/// `div_rust`. It still allocates because those APIs return Python lists.
pub fn elementwise_naive_rust(a: &[f32], b: &[f32], op: Op) -> Result<Vec<f32>, String> {
    validate_lengths(a, b)?;
    // Allocate once, then pass a mutable slice into the implementation. This
    // keeps ownership simple and mirrors how a Python result list is created.
    let mut out = vec![0.0; a.len()];
    naive_rust::naive_elementwise(a, b, &mut out, op);
    Ok(out)
}

/// Allocate a Python-list-style result and fill it through runtime dispatch.
///
/// This is the implementation behind `add`, `sub`, `mul`, and `div`. The
/// arithmetic may use SIMD, but list conversion and result allocation still
/// happen at the Python boundary.
pub fn elementwise_simd(a: &[f32], b: &[f32], op: Op) -> Result<Vec<f32>, String> {
    validate_lengths(a, b)?;
    let mut out = vec![0.0; a.len()];
    dispatch::dispatch_elementwise(a, b, &mut out, op);
    Ok(out)
}
