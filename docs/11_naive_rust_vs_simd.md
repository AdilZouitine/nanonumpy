# Naive Rust vs SIMD

The naive Rust implementation is a plain loop over slices. It is useful because it shows what happens when Python calls compiled code without explicit SIMD intrinsics.

The SIMD implementation makes vector width explicit. AVX2 processes 8 `f32` lanes per chunk, while SSE and NEON process 4 lanes per chunk. Tail elements go back through scalar code.

There is an important caveat: LLVM may auto-vectorize simple naive Rust loops in optimized builds. That means "naive Rust" describes the source code, not necessarily every instruction in the final binary.

Explicit SIMD is still useful for education because the code names the target features, vector loads, arithmetic intrinsics, and stores directly. It makes the CPU execution model visible.

SIMD is not always faster. Python conversion, memory bandwidth, and small input sizes can dominate.
