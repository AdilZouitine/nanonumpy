//! Shared operation types, validation, and allocation.

use crate::dispatch;
use crate::naive_rust;

#[derive(Clone, Copy, Debug)]
pub enum Op {
    Add,
    Sub,
    Mul,
    Div,
}

impl Op {
    pub fn apply(self, x: f32, y: f32) -> f32 {
        match self {
            Self::Add => x + y,
            Self::Sub => x - y,
            Self::Mul => x * y,
            Self::Div => x / y,
        }
    }
}

pub fn validate_lengths(a: &[f32], b: &[f32]) -> Result<(), String> {
    if a.len() != b.len() {
        return Err("arrays must have the same length".to_string());
    }

    Ok(())
}

pub fn elementwise_naive_rust(a: &[f32], b: &[f32], op: Op) -> Result<Vec<f32>, String> {
    validate_lengths(a, b)?;
    let mut out = vec![0.0; a.len()];
    naive_rust::naive_elementwise(a, b, &mut out, op);
    Ok(out)
}

pub fn elementwise_simd(a: &[f32], b: &[f32], op: Op) -> Result<Vec<f32>, String> {
    validate_lengths(a, b)?;
    let mut out = vec![0.0; a.len()];
    dispatch::dispatch_elementwise(a, b, &mut out, op);
    Ok(out)
}
