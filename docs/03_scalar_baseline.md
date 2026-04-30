# Scalar baseline

The scalar implementation in `src/scalar.rs` is the reference implementation. It processes one `f32` from `a` and one `f32` from `b` at a time, then writes one output value.

```text
load a[i]
load b[i]
compute a[i] + b[i]
store out[i]
```

The SIMD implementations use this same logic for tail elements. If AVX2 processes chunks of 8 and the length is 17, AVX2 handles 16 values and scalar code handles the final value.

Keeping a scalar reference makes the tutorial easier to reason about. SIMD changes how many values each instruction handles, not the mathematical result.
