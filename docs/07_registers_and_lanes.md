# Registers and lanes

Scalar code works one value at a time:

```text
a[0] + b[0] -> out[0]
a[1] + b[1] -> out[1]
a[2] + b[2] -> out[2]
```

SIMD code groups values into lanes inside a vector register:

```text
register A: [a0, a1, a2, a3]
register B: [b0, b1, b2, b3]
add        [a0+b0, a1+b1, a2+b2, a3+b3]
```

For AVX2:

```text
YMM register, 256 bits:
[f32, f32, f32, f32, f32, f32, f32, f32]
```

For SSE:

```text
XMM register, 128 bits:
[f32, f32, f32, f32]
```

For NEON:

```text
NEON vector register, 128 bits:
[f32, f32, f32, f32]
```

The compiler chooses the exact physical registers. The diagrams are conceptual, but they match the kind of vector work the instructions perform.
