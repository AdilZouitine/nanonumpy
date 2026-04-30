# ARM NEON

AArch64 includes NEON as part of the baseline architecture. NEON vector registers are 128 bits wide, so a `float32x4_t` value holds four `f32` lanes.

The implementation in `src/simd_arm.rs` uses:

- `vld1q_f32` to load 4 floats
- `vaddq_f32`, `vsubq_f32`, `vmulq_f32`, or `vdivq_f32` to compute 4 lanes
- `vst1q_f32` to store 4 floats

This applies to Linux aarch64 and macOS Apple Silicon. The loop advances by 4 floats at a time, then the scalar fallback handles any remaining tail elements.
