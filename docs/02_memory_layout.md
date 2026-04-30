# Memory layout

SIMD works best with contiguous memory. A Rust `Vec<f32>` stores 32-bit floats one after another:

```text
[a0][a1][a2][a3][a4][a5][a6][a7]
```

A slice, `&[f32]`, is a borrowed view into that memory. It carries a pointer and a length. SIMD intrinsics use raw pointers derived from slices so the CPU can load several adjacent values into a vector register.

Unaligned loads such as `_mm256_loadu_ps` do not require special alignment. They are convenient for a tutorial because Python list conversion gives us owned contiguous vectors, but we do not ask the allocator for a specific alignment.

Contiguous memory matters because a vector load can fetch adjacent lanes with one instruction. Strided or irregular memory would need gathers, scalar work, or copying into a contiguous buffer.
