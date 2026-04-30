# nano-numpy-simd

`nano-numpy-simd` is a tiny educational NumPy-like package written in Rust and exposed to Python.

It teaches one simple idea from several angles: **where does the loop run, and how many numbers does one CPU instruction process?**

The project implements 1D elementwise arithmetic on `f32` values:

- `add`
- `sub`
- `mul`
- `div`

It is intentionally small. The goal is not to replace NumPy. The goal is to help Python developers understand PyO3, maturin, FFI, Rust loops, SIMD dispatch, CPU registers, and benchmark pitfalls.

```text
                 nano-numpy-simd learning map

       Python list API                      buffer API
  easy to read, copies data          faster, uses preallocated output
             |                                    |
             v                                    v
   +-------------------+              +----------------------+
   | Python functions  |              | NumPy float32 arrays |
   | add_py, sub_py... |              | add_into(..., out)   |
   +-------------------+              +----------------------+
             |                                    |
             +----------------+-------------------+
                              |
                              v
                    +------------------+
                    | PyO3 Rust module |
                    | src/lib.rs       |
                    +------------------+
                              |
                              v
                    +------------------+
                    | validation + ops |
                    | src/ops.rs       |
                    +------------------+
                              |
                              v
                    +------------------+
                    | runtime dispatch |
                    | src/dispatch.rs  |
                    +------------------+
                    /        |         \
                   v         v          v
              scalar      x86_64      aarch64
             fallback   AVX2 / SSE     NEON
```

## Quick Start

Use `uv` for Python dependency management and `maturin` to build the Rust extension:

```bash
uv sync --extra dev
uv run maturin develop --release
uv run python python/examples/basic_usage.py
```

Expected output:

```python
[11.0, 22.0, 33.0, 44.0]
[11.0, 22.0, 33.0, 44.0]
[11.0, 22.0, 33.0, 44.0]
```

Always use `--release` before benchmarking. Debug builds are useful for development, but their timings are not meaningful.

## The Three Learning Levels

The same mathematical operation is implemented three ways:

```text
Level 0: Pure Python

    Python objects
    Python interpreter
    one high-level operation at a time

Level 1: Naive Rust

    Python calls Rust
    Rust receives Vec<f32>
    compiled loop over one f32 at a time

Level 2: Rust SIMD

    Python calls Rust
    Rust receives contiguous f32 memory
    one vector instruction processes several lanes
```

| Level | Function | Source | What it teaches |
|---|---|---|---|
| 0 | `add_py(a, b)` | [`nano_numpy_simd/pure_python.py`](nano_numpy_simd/pure_python.py) | Python interpreter baseline |
| 1 | `add_rust(a, b)` | [`src/naive_rust.rs`](src/naive_rust.rs) | Python-to-Rust speedup |
| 2 | `add(a, b)` | [`src/dispatch.rs`](src/dispatch.rs) | SIMD dispatch, still using list conversion |
| 2 fast path | `add_into(a, b, out)` | [`src/lib.rs`](src/lib.rs) | Buffer protocol, preallocated output |
| 3 | NumPy | external library | Mature optimized arrays |

## Clickable Reading Path

If you are a Python developer new to Rust extensions, read the code in this order:

1. [`python/examples/basic_usage.py`](python/examples/basic_usage.py) shows the public API.
2. [`nano_numpy_simd/__init__.py`](nano_numpy_simd/__init__.py) re-exports Python and Rust functions.
3. [`nano_numpy_simd/pure_python.py`](nano_numpy_simd/pure_python.py) shows the pure Python baseline.
4. [`src/lib.rs`](src/lib.rs) is the PyO3 boundary where Python calls enter Rust.
5. [`src/ops.rs`](src/ops.rs) validates input lengths and allocates list-style results.
6. [`src/naive_rust.rs`](src/naive_rust.rs) shows a plain compiled Rust loop.
7. [`src/scalar.rs`](src/scalar.rs) is the portable reference implementation.
8. [`src/dispatch.rs`](src/dispatch.rs) chooses AVX2, SSE, NEON, or scalar.
9. [`src/simd_x86.rs`](src/simd_x86.rs) contains x86_64 AVX2/SSE intrinsics.
10. [`src/simd_arm.rs`](src/simd_arm.rs) contains aarch64 NEON intrinsics.
11. [`python/examples/buffer_protocol_benchmark.py`](python/examples/buffer_protocol_benchmark.py) shows why preallocated buffers matter.
12. [`tests/test_buffer_api.py`](tests/test_buffer_api.py) documents the fast-path safety rules with executable examples.

## Basic Python List API

The list API is the easiest place to start because it looks like normal Python:

```python
import nano_numpy_simd as nn

a = [1.0, 2.0, 3.0, 4.0]
b = [10.0, 20.0, 30.0, 40.0]

print(nn.add_py(a, b))    # pure Python
print(nn.add_rust(a, b))  # naive Rust loop
print(nn.add(a, b))       # SIMD-dispatched Rust
```

The public list functions are:

```text
add_py      add_rust      add
sub_py      sub_rust      sub
mul_py      mul_rust      mul
div_py      div_rust      div
```

All list functions validate equal lengths, handle empty arrays, and return Python lists.

When you call `nn.add(a, b)`, follow this path:

```text
Python code
  |
  | nn.add(a, b)
  v
nano_numpy_simd/__init__.py
  |
  | re-exported from compiled module
  v
src/lib.rs
  |
  | #[pyfunction] fn add(...)
  v
src/ops.rs
  |
  | elementwise_simd(...)
  v
src/dispatch.rs
  |
  | choose platform implementation
  v
src/simd_x86.rs or src/simd_arm.rs or src/scalar.rs
  |
  | write result values
  v
Python receives list
```

Clickable version of that path:

- Python import surface: [`nano_numpy_simd/__init__.py`](nano_numpy_simd/__init__.py)
- Rust `add` binding: [`src/lib.rs`](src/lib.rs)
- shared validation and allocation: [`src/ops.rs`](src/ops.rs)
- runtime feature selection: [`src/dispatch.rs`](src/dispatch.rs)
- x86_64 SIMD code: [`src/simd_x86.rs`](src/simd_x86.rs)
- aarch64 SIMD code: [`src/simd_arm.rs`](src/simd_arm.rs)
- scalar fallback: [`src/scalar.rs`](src/scalar.rs)

## Faster Buffer API

The list API copies data:

```text
Python list -> Rust Vec<f32> -> compute -> Python list
```

That copy is great for learning, but it hides SIMD performance. The faster API accepts Python buffers, such as NumPy `float32` arrays, and writes into a preallocated output array:

```python
import numpy as np
import nano_numpy_simd as nn

a = np.arange(1_000_000, dtype=np.float32)
b = np.arange(1_000_000, dtype=np.float32)
out = np.empty_like(a)

nn.add_into(a, b, out)
```

The buffer functions are:

```text
add_into(a, b, out)
sub_into(a, b, out)
mul_into(a, b, out)
div_into(a, b, out)
```

They require:

- one-dimensional buffers
- C-contiguous layout
- `float32` / Rust `f32` values
- writable `out`
- no overlap between `out` and either input

The buffer API avoids Python list conversion and output-list allocation:

```text
             list API                         buffer API

  [Python list objects]              [NumPy float32 memory]
          |                                   |
          | copy each value                   | borrow buffer
          v                                   v
      Vec<f32>                         &[f32], &mut [f32]
          |                                   |
          | allocate result Vec               | write into out
          v                                   v
    Python result list                  existing output array
```

## What FFI Means

FFI means **Foreign Function Interface**. It is the mechanism that lets code written in one language call code written in another language.

Python cannot directly call an ordinary Rust function because Rust has its own calling conventions, type system, ownership rules, name mangling, and memory model. PyO3 creates the CPython-compatible wrapper.

```text
Python code
    |
    | import nano_numpy_simd
    v
CPython import machinery
    |
    | loads compiled extension module
    v
PyO3 module initializer
    |
    | registers #[pyfunction] functions
    v
Python calls nn.add(a, b)
    |
    | PyO3 extracts Python values
    v
Rust receives Vec<f32> or PyBuffer<f32>
    |
    | compute in Rust
    v
PyO3 converts the result or returns None
    |
    v
Python continues
```

In this project:

- `#[pymodule]` in [`src/lib.rs`](src/lib.rs) creates the Python extension module.
- `#[pyfunction]` exposes Rust functions to Python.
- maturin builds the Rust crate into a Python-importable shared library.
- PyO3 converts Python exceptions from Rust errors, such as length mismatch.

Python has several ways to call native code:

- CPython extension modules: what this project uses through PyO3.
- `ctypes`: Python's standard library can call C-compatible shared-library functions.
- CFFI: Python can call C-compatible APIs using C-like declarations.

PyO3 is the ergonomic choice here because Python users can import the result as a normal package:

```python
import nano_numpy_simd as nn
```

Under the hood, maturin builds a shared library with a Python-compatible name and installs it into the active `uv` environment. That is what this command does:

```bash
uv run maturin develop --release
```

### Ownership At The Boundary

The list API and buffer API have different ownership stories:

```text
list API

Python owns list
      |
      | PyO3 copies values
      v
Rust owns Vec<f32>
      |
      | Rust computes
      v
Rust owns result Vec<f32>
      |
      | PyO3 converts result
      v
Python owns result list
```

```text
buffer API

Python owns NumPy arrays
      |
      | PyO3 borrows exported buffers
      v
Rust sees &[f32] and &mut [f32]
      |
      | Rust computes in place
      v
Python still owns the same output array
```

The buffer API is faster because it avoids per-element conversion and avoids allocating a Python result list. It is also stricter because Rust must not create a mutable output slice that aliases an input slice.

The buffer API path is:

```text
Python code
  |
  | nn.add_into(a_np, b_np, out_np)
  v
src/lib.rs
  |
  | PyBuffer<f32> extracts borrowed buffers
  v
validate_buffer_inputs(...)
  |
  | check ndim, dtype, contiguity, length, writable out, no overlap
  v
Rust slices
  |
  | &[f32], &[f32], &mut [f32]
  v
same SIMD dispatcher as list API
  |
  | writes into out_np memory
  v
Python keeps using out_np
```

## Memory Layout

SIMD wants contiguous memory. A `Vec<f32>` or contiguous NumPy `float32` array stores values side by side:

```text
address grows left to right

base
 |
 v
+------+------+------+------+------+------+------+------+
| a[0] | a[1] | a[2] | a[3] | a[4] | a[5] | a[6] | a[7] |
+------+------+------+------+------+------+------+------+
   4B     4B     4B     4B     4B     4B     4B     4B
```

A Rust slice, `&[f32]`, is a borrowed view:

```text
&[f32]
+----------+--------+
| pointer  | length |
+----------+--------+
     |
     v
  contiguous f32 memory
```

The SIMD code uses raw pointers derived from slices:

```text
a.as_ptr().add(i)       -> address of a[i]
b.as_ptr().add(i)       -> address of b[i]
out.as_mut_ptr().add(i) -> address of out[i]
```

## Scalar Loop

Scalar code processes one element per loop iteration:

```text
i = 0: a[0] + b[0] -> out[0]
i = 1: a[1] + b[1] -> out[1]
i = 2: a[2] + b[2] -> out[2]
i = 3: a[3] + b[3] -> out[3]
```

That is the reference implementation in [`src/scalar.rs`](src/scalar.rs). SIMD code uses the same scalar function for tail values that do not fill a complete vector register.

The naive Rust implementation in [`src/naive_rust.rs`](src/naive_rust.rs) is intentionally close to the Python mental model:

```text
for i in 0..len:
    out[i] = a[i] + b[i]
```

The important difference is that Rust is working with raw `f32` values in contiguous memory, not Python float objects. Python list iteration has interpreter overhead, dynamic objects, and reference counting. Rust's loop is compiled to machine code.

One caveat: in release mode, LLVM may auto-vectorize simple loops. So "naive Rust" means the source code is naive. It does not guarantee that the final assembly contains no vector instructions.

## SIMD In One Picture

SIMD means **Single Instruction, Multiple Data**. One instruction operates on several lanes.

```text
Scalar:

  a0 + b0 -> out0
  a1 + b1 -> out1
  a2 + b2 -> out2
  a3 + b3 -> out3

SIMD:

  register A: [ a0 | a1 | a2 | a3 ]
  register B: [ b0 | b1 | b2 | b3 ]
       add:   [ +  | +  | +  | +  ]
  register C: [ c0 | c1 | c2 | c3 ]
```

For AVX2 with `f32`, the register has 8 lanes:

```text
YMM register, 256 bits

+-----+-----+-----+-----+-----+-----+-----+-----+
| f32 | f32 | f32 | f32 | f32 | f32 | f32 | f32 |
+-----+-----+-----+-----+-----+-----+-----+-----+
 lane0 lane1 lane2 lane3 lane4 lane5 lane6 lane7
```

For SSE and NEON, the register has 4 `f32` lanes:

```text
XMM / NEON register, 128 bits

+-----+-----+-----+-----+
| f32 | f32 | f32 | f32 |
+-----+-----+-----+-----+
 lane0 lane1 lane2 lane3
```

## Runtime Dispatch

The Rust code must not execute instructions unsupported by the current CPU. That can crash the process with an illegal instruction.

[`src/dispatch.rs`](src/dispatch.rs) chooses the implementation:

```text
                          dispatch_elementwise
                                  |
          +-----------------------+------------------------+
          |                                                |
      target_arch = x86_64                         target_arch = aarch64
          |                                                |
          v                                                v
  is AVX2 available?                                  use NEON
          |
   +------+------+
   |             |
  yes            no
   |             |
   v             v
 AVX2      is SSE available?
                 |
          +------+------+
          |             |
         yes            no
          |             |
          v             v
         SSE          scalar
```

On x86_64, the code uses runtime checks such as:

```rust
std::is_x86_feature_detected!("avx2")
```

On aarch64, NEON is part of the baseline architecture, so the NEON path is compiled for that target.

The dispatch layer is small but important. The Python function `nn.add(a, b)` does not know what CPU it is running on. The Rust dispatcher checks the target and available features, then calls the best supported implementation.

## AVX2 Loop Shape

AVX2 processes 8 `f32` values per iteration:

```text
while i + 8 <= len:

  load a[i..i+8]  -> ymm0
  load b[i..i+8]  -> ymm1
  add lanes       -> ymm2
  store ymm2      -> out[i..i+8]

  i += 8

tail:
  scalar handles remaining values
```

Conceptual x86_64 assembly for add:

```asm
vmovups ymm0, [a + i * 4]    ; load 8 f32 values
vmovups ymm1, [b + i * 4]    ; load 8 f32 values
vaddps  ymm2, ymm0, ymm1     ; add 8 lanes
vmovups [out + i * 4], ymm2  ; store 8 f32 values
```

The variables in Rust are not guaranteed to become exactly `ymm0`, `ymm1`, and `ymm2`. The compiler decides final register allocation. The assembly above is conceptual, but it matches the operation.

Read the real implementation in [`src/simd_x86.rs`](src/simd_x86.rs). Look for:

- `_mm256_loadu_ps`: load 8 contiguous `f32` values
- `_mm256_add_ps`: add 8 lanes
- `_mm256_sub_ps`: subtract 8 lanes
- `_mm256_mul_ps`: multiply 8 lanes
- `_mm256_div_ps`: divide 8 lanes
- `_mm256_storeu_ps`: store 8 contiguous `f32` results

SSE in the same file uses 128-bit XMM registers and processes 4 `f32` values per iteration. That gives the tutorial a smaller x86_64 fallback before scalar code.

## NEON Loop Shape

NEON processes 4 `f32` values per iteration:

```text
while i + 4 <= len:

  load a[i..i+4]  -> q0
  load b[i..i+4]  -> q1
  add lanes       -> q2
  store q2        -> out[i..i+4]

  i += 4
```

Conceptual AArch64 assembly:

```asm
ldr  q0, [a]
ldr  q1, [b]
fadd v2.4s, v0.4s, v1.4s
str  q2, [out]
```

Read the real implementation in [`src/simd_arm.rs`](src/simd_arm.rs). Look for:

- `vld1q_f32`: load 4 contiguous `f32` values
- `vaddq_f32`: add 4 lanes
- `vsubq_f32`: subtract 4 lanes
- `vmulq_f32`: multiply 4 lanes
- `vdivq_f32`: divide 4 lanes on AArch64
- `vst1q_f32`: store 4 contiguous `f32` results

This project does not fake NEON division with reciprocal approximations. It uses the stable AArch64 intrinsic.

## Inspecting Generated Assembly

The intrinsic code is not the final machine code. The compiler lowers intrinsics, schedules instructions, allocates registers, and may inline functions.

To ask Rust to emit assembly:

```bash
cargo rustc --release -- --emit=asm
```

Or install `cargo-asm`:

```bash
cargo install cargo-asm
cargo asm nano_numpy_simd::simd_x86::avx2_elementwise
```

Always inspect release builds. Debug builds contain extra checks and are not representative of optimized code.

## Tail Handling

Not every array length is divisible by 8 or 4. The SIMD loop handles full vector chunks, then scalar code handles the rest.

```text
len = 17, AVX2 width = 8

indexes:  0 1 2 3 4 5 6 7 | 8 9 10 11 12 13 14 15 | 16
          <---- AVX2 ----> | <------- AVX2 -------> | scalar tail
```

The tests include lengths `0, 1, 3, 4, 5, 7, 8, 9, 15, 16, 17` to exercise both vector chunks and tails.

## What Happens Inside The CPU?

At a simplified level, the CPU does this:

```text
             memory
               |
               | load
               v
        +--------------+
        | vector regs  |  XMM / YMM / NEON Q
        +--------------+
               |
               | SIMD arithmetic
               v
        +--------------+
        | vector units |
        +--------------+
               |
               | store
               v
             memory
```

Modern CPUs are pipelined. Loads, arithmetic, stores, and branch prediction overlap. SIMD helps because it reduces loop overhead and increases arithmetic throughput, but it does not remove memory bandwidth limits or Python/Rust conversion costs.

A beginner-friendly CPU model:

```text
Python calls Rust once
        |
        v
Rust loop starts
        |
        +--> load unit reads contiguous f32 values
        |
        +--> vector register holds several lanes
        |
        +--> vector execution unit performs add/sub/mul/div
        |
        +--> store unit writes output values
        |
        v
loop branch decides whether another chunk remains
```

Performance can be limited by:

- Python conversion overhead
- memory bandwidth
- cache misses
- alignment
- CPU frequency changes
- thermal throttling
- compiler optimizations
- small input sizes

This is why the tutorial compares several paths instead of claiming that SIMD always wins.

## Why The List Benchmark Is Only A Little Faster

The list API benchmark includes more than arithmetic:

```text
Python list input
      |
      | convert every Python float
      v
Rust Vec<f32>
      |
      | SIMD compute
      v
Rust Vec<f32>
      |
      | convert every f32 back
      v
Python list output
```

For large lists, most time can be spent converting and allocating. That is why `add(a, b)` may only be around 1.5x faster than pure Python for list inputs.

The buffer API removes those costs:

```text
NumPy float32 input buffers
      |
      | borrow memory, no element conversion
      v
Rust &[f32]
      |
      | SIMD compute
      v
Rust &mut [f32]
      |
      | writes into existing NumPy output
      v
same Python output array
```

On one Apple Silicon run with 5,000,000 values:

```text
Implementation       Time          Speedup vs Python
Pure Python list     180.690 ms       1.00x
Rust list API        126.435 ms       1.43x
Rust buffer into       0.785 ms     230.13x
NumPy out=             0.770 ms     234.65x
```

These numbers are examples from one machine. Run the benchmark locally.

## Running The Benchmarks

```bash
uv sync --extra dev
uv run maturin develop --release
uv run python python/examples/benchmark.py
uv run python python/examples/buffer_protocol_benchmark.py
uv run python python/examples/full_performance_comparison.py
```

For Rust-only benchmarking:

```bash
cargo bench
```

Benchmarks are noisy. CPU frequency, cache state, thermal throttling, background processes, compiler version, and input size all matter.

Why NumPy may still win:

- it works directly with array buffers
- it has years of low-level optimization
- it avoids Python per-element overhead
- it may use platform-specific kernels
- it has mature handling for shapes, strides, and dtypes

Why this Rust tutorial can get close for `add_into`:

- the input is already contiguous `float32`
- the output is preallocated
- the operation is simple
- Rust dispatches to SIMD
- there is no Python per-element loop

## Safety Notes

The SIMD code uses `unsafe` because it does things Rust cannot prove safe by itself:

- raw pointer arithmetic
- unaligned vector loads and stores
- target-specific CPU instructions
- mutable output slices from Python buffers

The project keeps the unsafe boundary small:

- lengths are validated before compute
- SIMD loops use `i + width <= len`
- tail elements use scalar code
- x86_64 SIMD is guarded by runtime feature detection
- buffer outputs must be writable and non-overlapping

## Source Map

| Concept | Source |
|---|---|
| Python exports | [`nano_numpy_simd/__init__.py`](nano_numpy_simd/__init__.py) |
| Pure Python baseline | [`nano_numpy_simd/pure_python.py`](nano_numpy_simd/pure_python.py) |
| PyO3 list functions | [`src/lib.rs`](src/lib.rs) |
| PyO3 buffer functions | [`src/lib.rs`](src/lib.rs) |
| Shared operation validation | [`src/ops.rs`](src/ops.rs) |
| Naive Rust loop | [`src/naive_rust.rs`](src/naive_rust.rs) |
| Scalar fallback | [`src/scalar.rs`](src/scalar.rs) |
| Runtime dispatch | [`src/dispatch.rs`](src/dispatch.rs) |
| x86_64 AVX2/SSE | [`src/simd_x86.rs`](src/simd_x86.rs) |
| aarch64 NEON | [`src/simd_arm.rs`](src/simd_arm.rs) |
| FFI source notes | [`src/ffi_explain.rs`](src/ffi_explain.rs) |
| Assembly source notes | [`src/asm_explain.rs`](src/asm_explain.rs) |
| Basic usage example | [`python/examples/basic_usage.py`](python/examples/basic_usage.py) |
| List benchmark | [`python/examples/benchmark.py`](python/examples/benchmark.py) |
| Buffer benchmark | [`python/examples/buffer_protocol_benchmark.py`](python/examples/buffer_protocol_benchmark.py) |
| Full comparison | [`python/examples/full_performance_comparison.py`](python/examples/full_performance_comparison.py) |
| Basic tests | [`tests/test_basic.py`](tests/test_basic.py) |
| Error tests | [`tests/test_errors.py`](tests/test_errors.py) |
| NumPy comparison tests | [`tests/test_against_numpy.py`](tests/test_against_numpy.py) |
| Buffer API tests | [`tests/test_buffer_api.py`](tests/test_buffer_api.py) |
| Extra topic docs | [`docs/`](docs/) |

This README is the main tutorial. The files in `docs/` split the same ideas into smaller topic pages for readers who prefer shorter chapters.

## Topic Chapters

The README is intentionally self-contained, but each topic also has a shorter chapter:

- [Python to Rust](docs/01_python_to_rust.md)
- [Memory layout](docs/02_memory_layout.md)
- [Scalar baseline](docs/03_scalar_baseline.md)
- [x86_64 SIMD](docs/04_x86_simd.md)
- [ARM NEON](docs/05_arm_neon.md)
- [Dispatch](docs/06_dispatch.md)
- [Registers and lanes](docs/07_registers_and_lanes.md)
- [Benchmarking](docs/08_benchmarking.md)
- [FFI: how Python calls Rust](docs/09_ffi_python_to_rust.md)
- [Pure Python vs Rust](docs/10_pure_python_vs_rust.md)
- [Naive Rust vs SIMD](docs/11_naive_rust_vs_simd.md)
- [From Rust intrinsics to assembly](docs/12_from_intrinsics_to_assembly.md)
- [What happens inside the CPU?](docs/13_cpu_execution_model.md)

## Build And Test

```bash
uv sync --extra dev
uv run maturin develop --release
uv run pytest
cargo test
cargo clippy -- -D warnings
cargo fmt --check
```

If your local shell has both `VIRTUAL_ENV` and `CONDA_PREFIX` set, maturin may ask you to unset one. This local workaround is often enough:

```bash
env -u CONDA_PREFIX uv run maturin develop --release
```

## Platform Notes

- Linux x86_64: scalar plus x86 SIMD.
- macOS Intel: scalar plus x86 SIMD.
- Linux aarch64: scalar plus NEON.
- macOS Apple Silicon: scalar plus NEON.

The tutorial uses `x86_64`, not `x84`.

## Important Caveats

The naive Rust baseline is source-level naive Rust. In optimized builds, LLVM may auto-vectorize simple loops. This is useful to discuss because modern compilers are smart. The explicit SIMD implementation is still valuable because it shows vector registers and instructions directly.

Do not assume SIMD is always faster. Python-to-Rust conversion overhead can dominate for small arrays, and simple arithmetic can be limited by memory bandwidth. SIMD benefits become clearer when data is already contiguous and the output is preallocated.

## Limitations

- Only `f32` is supported.
- Only 1D arrays are supported.
- No broadcasting.
- No shapes or strides beyond requiring simple contiguous buffers.
- No direct alignment experiments yet.
- No wheel build matrix yet.
- Benchmark results depend heavily on CPU, compiler, and input size.
- Naive Rust source code may still be auto-vectorized by LLVM in release builds.

## Suggested Next Improvements

- support `f64`
- add shape-aware APIs
- add broadcasting experiments
- add wheels with cibuildwheel
- add more operations
- add multithreading with Rayon
- add alignment experiments
- add explicit benchmark plots
- add assembly snapshots for different CPUs
- add runtime feature printout, such as "using AVX2", "using NEON", or "using scalar fallback"
