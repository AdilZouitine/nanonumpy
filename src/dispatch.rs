//! Runtime SIMD dispatch.

use crate::ops::Op;

pub fn dispatch_elementwise(a: &[f32], b: &[f32], out: &mut [f32], op: Op) {
    dispatch_platform(a, b, out, op);
}

#[cfg(target_arch = "x86_64")]
fn dispatch_platform(a: &[f32], b: &[f32], out: &mut [f32], op: Op) {
    if std::is_x86_feature_detected!("avx2") {
        // SAFETY: Runtime detection above ensures this process may execute AVX2 instructions.
        unsafe {
            crate::simd_x86::avx2_elementwise(a, b, out, op);
        }
        return;
    }

    if std::is_x86_feature_detected!("sse") {
        // SAFETY: Runtime detection above ensures this process may execute SSE instructions.
        unsafe {
            crate::simd_x86::sse_elementwise(a, b, out, op);
        }
        return;
    }

    crate::scalar::scalar_elementwise(a, b, out, op);
}

#[cfg(target_arch = "aarch64")]
fn dispatch_platform(a: &[f32], b: &[f32], out: &mut [f32], op: Op) {
    // NEON is part of the baseline AArch64 architecture.
    crate::simd_arm::neon_elementwise(a, b, out, op);
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
fn dispatch_platform(a: &[f32], b: &[f32], out: &mut [f32], op: Op) {
    crate::scalar::scalar_elementwise(a, b, out, op);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dispatch_handles_non_vector_length() {
        let a = vec![1.0; 17];
        let b = vec![2.0; 17];
        let mut out = vec![0.0; 17];

        dispatch_elementwise(&a, &b, &mut out, Op::Add);

        assert_eq!(out, vec![3.0; 17]);
    }
}
