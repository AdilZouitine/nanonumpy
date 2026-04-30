//! x86_64 SIMD implementations using stable `std::arch` intrinsics.

use crate::ops::Op;
use crate::scalar;
use std::arch::x86_64::*;

#[target_feature(enable = "avx2")]
pub unsafe fn avx2_elementwise(a: &[f32], b: &[f32], out: &mut [f32], op: Op) {
    let len = a.len();
    let mut i = 0;

    while i + 8 <= len {
        // SAFETY: `i + 8 <= len` guarantees the 8-lane loads and store stay in bounds.
        unsafe {
            // Load 8 contiguous f32 values from a into a 256-bit AVX register.
            // Conceptually this becomes something like:
            // vmovups ymm0, [a_ptr + i * 4]
            let va = _mm256_loadu_ps(a.as_ptr().add(i));
            let vb = _mm256_loadu_ps(b.as_ptr().add(i));

            // Add/sub/mul/div 8 f32 lanes at once in YMM registers.
            // Conceptual add:
            // vaddps ymm2, ymm0, ymm1
            let vc = match op {
                Op::Add => _mm256_add_ps(va, vb),
                Op::Sub => _mm256_sub_ps(va, vb),
                Op::Mul => _mm256_mul_ps(va, vb),
                Op::Div => _mm256_div_ps(va, vb),
            };

            _mm256_storeu_ps(out.as_mut_ptr().add(i), vc);
        }
        i += 8;
    }

    scalar::scalar_elementwise(&a[i..], &b[i..], &mut out[i..], op);
}

#[target_feature(enable = "sse")]
pub unsafe fn sse_elementwise(a: &[f32], b: &[f32], out: &mut [f32], op: Op) {
    let len = a.len();
    let mut i = 0;

    while i + 4 <= len {
        // SAFETY: `i + 4 <= len` guarantees the 4-lane loads and store stay in bounds.
        unsafe {
            // Load 4 contiguous f32 values into 128-bit XMM registers.
            // Conceptually:
            // movups xmm0, [a_ptr + i * 4]
            let va = _mm_loadu_ps(a.as_ptr().add(i));
            let vb = _mm_loadu_ps(b.as_ptr().add(i));

            let vc = match op {
                Op::Add => _mm_add_ps(va, vb),
                Op::Sub => _mm_sub_ps(va, vb),
                Op::Mul => _mm_mul_ps(va, vb),
                Op::Div => _mm_div_ps(va, vb),
            };

            _mm_storeu_ps(out.as_mut_ptr().add(i), vc);
        }
        i += 4;
    }

    scalar::scalar_elementwise(&a[i..], &b[i..], &mut out[i..], op);
}
