# x86_64 SIMD

x86_64 CPUs commonly expose 128-bit XMM registers for SSE and 256-bit YMM registers for AVX/AVX2. For `f32`, an XMM register holds 4 lanes and a YMM register holds 8 lanes.

The implementation in `src/simd_x86.rs` uses this pattern:

```text
load vector from a
load vector from b
compute add/sub/mul/div across lanes
store vector to out
```

AVX2 uses `_mm256_loadu_ps`, `_mm256_add_ps`, and `_mm256_storeu_ps` for addition. SSE uses the 128-bit forms: `_mm_loadu_ps`, `_mm_add_ps`, and `_mm_storeu_ps`.

The code calls AVX2 only after `is_x86_feature_detected!("avx2")` succeeds. Unsupported SIMD instructions can crash the process, so runtime detection is part of the safety story.
