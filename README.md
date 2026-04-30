# nano-numpy-simd

This is a tiny educational NumPy-like package written in Rust and exposed to Python.

The project focuses on:

- elementwise arithmetic
- Python/Rust FFI
- scalar Rust
- naive Rust
- SIMD Rust
- CPU registers and vector instructions

It is a tutorial repository, not a production NumPy clone.

## Quick Start

```bash
uv sync --extra dev
uv run maturin develop --release
uv run python python/examples/basic_usage.py
```

## Basic Usage

```python
import nano_numpy_simd as nn

a = [1.0, 2.0, 3.0, 4.0]
b = [10.0, 20.0, 30.0, 40.0]

print(nn.add_py(a, b))
print(nn.add_rust(a, b))
print(nn.add(a, b))
```

Expected output:

```python
[11.0, 22.0, 33.0, 44.0]
[11.0, 22.0, 33.0, 44.0]
[11.0, 22.0, 33.0, 44.0]
```

## Architecture

```text
Python list
   ↓
Pure Python baseline, or PyO3 binding in src/lib.rs
   ↓
Rust operation layer in src/ops.rs
   ↓
naive Rust loop or SIMD dispatcher
   ↓
SIMD implementation if available
   ↓
scalar fallback otherwise
   ↓
Python result
```

## Learning Path

1. Run `python/examples/basic_usage.py`
2. Read `nano_numpy_simd/pure_python.py`
3. Read `src/naive_rust.rs`
4. Read `src/scalar.rs`
5. Read `src/dispatch.rs`
6. Read `src/simd_x86.rs` or `src/simd_arm.rs`
7. Read `docs/09_ffi_python_to_rust.md`
8. Read `docs/12_from_intrinsics_to_assembly.md`
9. Read `docs/13_cpu_execution_model.md`
10. Run `python/examples/full_performance_comparison.py`

## Performance Levels

```text
Level 0: Pure Python
    for i in range(len(a)):
        out[i] = a[i] + b[i]

Level 1: Naive Rust
    compiled loop over f32 slices

Level 2: Rust SIMD
    load vector register
    add/sub/mul/div several lanes
    store vector register
    handle tail
```

| Level | Function | Where | What it teaches |
|---|---|---|---|
| 0 | `add_py` | `nano_numpy_simd/pure_python.py` | Python interpreter baseline |
| 1 | `add_rust` | `src/naive_rust.rs` | Compiled Rust loop |
| 2 | `add` | `src/dispatch.rs` + SIMD files | Runtime SIMD dispatch |
| 3 | NumPy | external library | Mature optimized array library |

## Clickable Source Map

| Concept | File | What to look at |
|---|---|---|
| Python-facing API | `src/lib.rs` | `#[pyfunction] add`, `sub`, `mul`, `div` |
| Pure Python baseline | `nano_numpy_simd/pure_python.py` | list comprehensions |
| Naive Rust baseline | `src/naive_rust.rs` | plain compiled loop |
| Input validation | `src/ops.rs` | length checks |
| Scalar fallback | `src/scalar.rs` | plain Rust loop |
| CPU dispatch | `src/dispatch.rs` | runtime feature selection |
| x86 SIMD | `src/simd_x86.rs` | AVX2/SSE implementation |
| ARM SIMD | `src/simd_arm.rs` | NEON implementation |
| FFI explanation | `docs/09_ffi_python_to_rust.md` | Python calling Rust |
| Assembly explanation | `docs/12_from_intrinsics_to_assembly.md` | intrinsics to assembly |
| CPU execution | `docs/13_cpu_execution_model.md` | registers, lanes, pipelines |
| Register explanation | `docs/07_registers_and_lanes.md` | lanes, registers, vector width |

## Core Principle

Each implementation computes the same mathematical result. The difference is where the loop runs and how much work each machine instruction performs. Pure Python performs one high-level interpreted operation at a time. Naive Rust performs one compiled low-level operation at a time. SIMD Rust performs several low-level operations per vector instruction.

SIMD means Single Instruction, Multiple Data. Instead of adding one `f32` at a time, the CPU can load several `f32` values into a vector register and apply one instruction to all lanes at once. For example, an AVX register can hold eight `f32` values. A NEON register commonly holds four `f32` values. The loop processes chunks with SIMD and then handles the remaining tail elements with scalar code.

## `add(a, b)` Walkthrough

Python calls `nn.add(a, b)`. CPython routes the call into the PyO3 extension module. PyO3 receives Python lists and converts them to Rust vectors. Rust validates equal lengths, then the dispatcher chooses AVX2, SSE, NEON, or scalar. The SIMD loop handles chunks, scalar code handles leftovers, and PyO3 converts the Rust `Vec<f32>` back into a Python list.

## Which Registers Are Used?

- x86_64 AVX uses 256-bit YMM registers, often 8 lanes of `f32`.
- x86_64 SSE uses 128-bit XMM registers, often 4 lanes of `f32`.
- ARM NEON uses 128-bit vector registers, often 4 lanes of `f32`.
- The exact physical register allocation is decided by the compiler, but the intrinsics map to vector register operations.

```text
a_vec = [a0, a1, a2, a3, a4, a5, a6, a7]
b_vec = [b0, b1, b2, b3, b4, b5, b6, b7]
out   = [a0+b0, a1+b1, ..., a7+b7]
```

## Safety Notes

The SIMD code uses `unsafe` because it performs raw pointer arithmetic, unaligned vector loads and stores, and target-specific instructions. Bounds are handled by chunk loops such as `i + 8 <= len`, and target instructions are called only after compile-time `cfg` checks plus runtime CPU feature detection where needed.

## Build And Test

```bash
uv sync --extra dev
uv run maturin develop --release
uv run pytest
uv run python python/examples/benchmark.py
cargo test
cargo clippy -- -D warnings
cargo fmt --check
```

## Platform Notes

- Linux x86_64: scalar plus x86 SIMD.
- macOS Intel: scalar plus x86 SIMD.
- Linux aarch64: scalar plus NEON.
- macOS Apple Silicon: scalar plus NEON.

## Important Warning About Naive Rust

The naive Rust baseline is source-level naive Rust. In optimized builds, LLVM may auto-vectorize simple loops. This is useful to discuss because modern compilers are smart. The explicit SIMD implementation is included because it shows the registers and instructions directly.

## Benchmark Warning

Do not pretend that SIMD is always faster. Python-to-Rust conversion overhead can dominate for small arrays. SIMD benefits become clearer when arrays are large and already contiguous.

Always build with `uv run maturin develop --release` before benchmarking. Debug builds include extra checks and skip important optimizations, so their benchmark numbers are not meaningful.

## Limitations

- Python list conversion causes copies.
- Only `f32` is supported.
- Only 1D arrays are supported.
- No broadcasting.
- No shapes or strides.
- No memory alignment experiments yet.
- No direct NumPy buffer protocol yet.
- Benchmark results depend heavily on CPU, compiler, and input size.
- Naive Rust source code may still be auto-vectorized by LLVM in release builds.

## Suggested Next Improvements

- support `f64`
- support NumPy array buffer protocol directly
- add Python buffer protocol support
- add wheels with cibuildwheel
- add more operations
- add multithreading with Rayon
- add alignment experiments
- add explicit benchmark plots
- add assembly snapshots for different CPUs
- add runtime feature printout, such as "using AVX2", "using NEON", or "using scalar fallback"
