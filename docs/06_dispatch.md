# Dispatch

Dispatch chooses which implementation runs for the current platform and CPU.

Compile-time `cfg` decides which files are even available. x86_64 builds compile `src/simd_x86.rs`; aarch64 builds compile `src/simd_arm.rs`.

Runtime feature detection matters on x86_64 because not every x86_64 CPU supports AVX2. `is_x86_feature_detected!("avx2")` checks whether the current CPU and operating system can safely execute AVX2 instructions.

The order is:

```text
x86_64: AVX2 if available, then SSE, then scalar
aarch64: NEON, then scalar tail inside the NEON loop
other: scalar
```

Calling unsupported SIMD instructions directly can cause an illegal instruction crash. That is why target-specific functions sit behind dispatch.
