# From Rust intrinsics to assembly

### 1. Intrinsics are not magic

A Rust intrinsic such as `_mm256_add_ps` represents a CPU vector operation. The compiler lowers the intrinsic into target-specific machine instructions.

### 2. Example: AVX2 addition

Rust intrinsic:

```rust
let va = _mm256_loadu_ps(a.as_ptr().add(i));
let vb = _mm256_loadu_ps(b.as_ptr().add(i));
let vc = _mm256_add_ps(va, vb);
_mm256_storeu_ps(out.as_mut_ptr().add(i), vc);
```

Conceptual x86_64 assembly:

```asm
vmovups ymm0, [rdi + 4*i]    ; load 8 f32 values from a
vmovups ymm1, [rsi + 4*i]    ; load 8 f32 values from b
vaddps  ymm2, ymm0, ymm1     ; add 8 lanes
vmovups [rdx + 4*i], ymm2    ; store 8 f32 results
```

`ymm0`, `ymm1`, and `ymm2` are 256-bit vector registers. Each YMM register can hold eight 32-bit floats. `vaddps` means vector add packed single-precision floats. The CPU performs eight additions conceptually from one instruction.

### 3. Example: SSE addition

Conceptual x86_64 assembly:

```asm
movups xmm0, [rdi + 4*i]
movups xmm1, [rsi + 4*i]
addps  xmm0, xmm1
movups [rdx + 4*i], xmm0
```

`xmm` registers are 128-bit registers. They hold four `f32` values, so SSE processes four lanes at a time.

### 4. Example: ARM NEON addition

Rust intrinsic:

```rust
let va = vld1q_f32(a.as_ptr().add(i));
let vb = vld1q_f32(b.as_ptr().add(i));
let vc = vaddq_f32(va, vb);
vst1q_f32(out.as_mut_ptr().add(i), vc);
```

Conceptual AArch64 assembly:

```asm
ldr q0, [x0]
ldr q1, [x1]
fadd v2.4s, v0.4s, v1.4s
str q2, [x2]
```

`q0`, `q1`, and `q2` are 128-bit vector registers. `v0.4s` means vector register 0 interpreted as four 32-bit single-precision float lanes. `fadd v2.4s, v0.4s, v1.4s` adds four lanes.

### 5. How to inspect the generated assembly

```bash
cargo rustc --release -- --emit=asm
```

Or install cargo-asm:

```bash
cargo install cargo-asm
cargo asm nano_numpy_simd::simd_x86::avx2_elementwise
```

Symbols may be optimized, inlined, or renamed.

### 6. Debug vs release

Always inspect release builds. Debug builds contain extra checks and are not representative of optimized code.

### 7. Intrinsics and register allocation

The code names variables like `va`, `vb`, and `vc`, but these are not guaranteed to map exactly to `ymm0`, `ymm1`, and `ymm2`. The compiler decides the final register allocation. The tutorial diagrams are conceptual but match the type of operation.
