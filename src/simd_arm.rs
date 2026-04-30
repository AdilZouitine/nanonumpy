//! AArch64 NEON implementation using stable `std::arch` intrinsics.

use crate::ops::Op;
use crate::scalar;
use std::arch::aarch64::*;

pub fn neon_elementwise(a: &[f32], b: &[f32], out: &mut [f32], op: Op) {
    let len = a.len();
    let mut i = 0;

    while i + 4 <= len {
        // SAFETY: `i + 4 <= len` keeps the 4-lane NEON loads and store in bounds.
        unsafe {
            // Load 4 contiguous f32 values into a 128-bit NEON register.
            // Conceptually:
            // ldr q0, [x0]
            let va = vld1q_f32(a.as_ptr().add(i));
            let vb = vld1q_f32(b.as_ptr().add(i));

            // Add/sub/mul/div 4 f32 lanes in ARM vector registers.
            // Conceptual add:
            // fadd v2.4s, v0.4s, v1.4s
            let vc = match op {
                Op::Add => vaddq_f32(va, vb),
                Op::Sub => vsubq_f32(va, vb),
                Op::Mul => vmulq_f32(va, vb),
                Op::Div => vdivq_f32(va, vb),
            };

            vst1q_f32(out.as_mut_ptr().add(i), vc);
        }
        i += 4;
    }

    scalar::scalar_elementwise(&a[i..], &b[i..], &mut out[i..], op);
}
