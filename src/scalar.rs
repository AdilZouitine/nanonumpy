//! Portable scalar fallback used as the reference implementation.
//!
//! Every platform can run this code. SIMD implementations call back into this
//! function for the "tail" elements left over after vector-width chunks.

use crate::ops::Op;

pub fn scalar_elementwise(a: &[f32], b: &[f32], out: &mut [f32], op: Op) {
    // `zip` walks the three slices together:
    //
    //   (a[0], b[0], out[0])
    //   (a[1], b[1], out[1])
    //   ...
    //
    // The caller validates equal lengths, so no element is dropped.
    for ((x, y), slot) in a.iter().zip(b).zip(out.iter_mut()) {
        // Store into the caller-owned output buffer instead of allocating here.
        *slot = op.apply(*x, *y);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn handles_tail_lengths() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0];
        let b = [10.0, 20.0, 30.0, 40.0, 50.0];
        let mut out = [0.0; 5];

        scalar_elementwise(&a, &b, &mut out, Op::Add);

        assert_eq!(out, [11.0, 22.0, 33.0, 44.0, 55.0]);
    }
}
