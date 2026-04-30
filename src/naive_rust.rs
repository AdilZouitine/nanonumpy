//! Source-level naive Rust implementation.
//!
//! This code intentionally avoids explicit SIMD intrinsics. In optimized builds,
//! LLVM may still auto-vectorize simple loops like this. For the tutorial, this
//! is "naive source code", not a guarantee about final assembly.

use crate::ops::Op;

pub fn naive_elementwise(a: &[f32], b: &[f32], out: &mut [f32], op: Op) {
    for i in 0..a.len() {
        out[i] = op.apply(a[i], b[i]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn multiplies_values() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        let mut out = [0.0; 3];

        naive_elementwise(&a, &b, &mut out, Op::Mul);

        assert_eq!(out, [4.0, 10.0, 18.0]);
    }
}
