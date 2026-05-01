# nano-numpy-simd

`nano-numpy-simd` is a tiny NumPy-like package written in Rust and exposed to Python.

It teaches one question from several angles: where does the loop run, and how many numbers does one CPU instruction process?

The project implements 1D elementwise arithmetic on `f32` values:

- `add`
- `sub`
- `mul`
- `div`

It is intentionally small. It does not try to replace NumPy. It gives Python developers a compact way to study PyO3, maturin, FFI, Rust loops, SIMD dispatch, CPU registers, and benchmark pitfalls.

## Table of contents

- Getting started
  - [Install the tools first](#install-the-tools-first)
  - [Why this matters](#why-this-matters)
  - [Quick start](#quick-start)
- Guide map
  - [Learning levels](#learning-levels)
  - [Clickable reading path](#clickable-reading-path)
- Tutorial implementation path
  - [Start here: Basic Python list API](#start-here-basic-python-list-api)
  - [Faster buffer API](#faster-buffer-api)
  - [What FFI means](#what-ffi-means)
  - [Memory layout](#memory-layout)
  - [Scalar loop](#scalar-loop)
- SIMD and CPU details
  - [SIMD in one picture](#simd-in-one-picture)
  - [Runtime dispatch](#runtime-dispatch)
  - [AVX2 loop shape](#avx2-loop-shape)
  - [NEON loop shape](#neon-loop-shape)
  - [Inspecting generated assembly](#inspecting-generated-assembly)
  - [Tail handling](#tail-handling)
  - [What happens inside the CPU?](#what-happens-inside-the-cpu)
- Validation and results
  - [Why the list benchmark is only a little faster](#why-the-list-benchmark-is-only-a-little-faster)
  - [Running the benchmarks](#running-the-benchmarks)
  - [Safety notes](#safety-notes)
- Going further
  - [What would be next to become a real NumPy](#what-would-be-next-to-become-a-real-numpy)

## Install the tools first

You need two toolchains:

- `uv` for Python environments and dependencies.
- Rust/Cargo for compiling the native extension.

Install `uv`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Install Rust and Cargo:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

Check both tools:

```bash
uv --version
cargo --version
rustc --version
```

## Why this matters

The fastest path in this tutorial is not the list API. It is the buffer API, where Python gives Rust direct access to contiguous `float32` memory and Rust writes into a preallocated output array.

On one Apple Silicon run with 5,000,000 values:

```text
Size: 5,000,000 float32 values
Implementation     Time           Speedup vs Python
Pure Python list      180.690 ms       1.00x
Rust list API         126.435 ms       1.43x
Rust buffer into        0.785 ms     230.13x
NumPy allocated         0.776 ms     232.82x
NumPy out =             0.770 ms     234.65x
```

The useful lesson for Python developers: calling Rust is not automatically fast if Python still has to convert millions of objects. The speedup shows up when data is already in a contiguous numeric buffer.

The rest of this README builds up to that result step by step. We start with normal Python lists, then look at how Python calls Rust, then look at memory layout, then finally explain why SIMD needs the data to have a very specific shape in RAM.

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

## Quick start

Use `uv` for Python dependency management and `maturin` to build the Rust extension:

```bash
uv sync --extra dev
uv run maturin develop --release
uv run python python/examples/basic_usage.py
```

Example inputs:

```python
a = [1.0, 2.0, 3.0, 4.0]
b = [10.0, 20.0, 30.0, 40.0]
```

Each implementation computes element-wise addition: `a[i] + b[i]`.

Expected output:

```python
[11.0, 22.0, 33.0, 44.0]
[11.0, 22.0, 33.0, 44.0]
[11.0, 22.0, 33.0, 44.0]
```

Always use `--release` before benchmarking. Debug builds are useful for development, but their timings are not meaningful.

## Learning levels

Before looking at registers or CPU instructions, keep the learning path simple: every implementation computes the same arithmetic. What changes is where the loop runs and what kind of data the loop sees.

Inside this repository, the same mathematical operation is implemented at three levels:

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
| 2 buffer path | `add_into(a, b, out)` | [`src/lib.rs`](src/lib.rs) | Buffer protocol, preallocated output |
| External baseline | NumPy | external library | Mature optimized arrays |

**This README is the main guide for the tutorial. The code files are linked from here so readers can jump directly from concept to implementation.**

## Clickable reading path

The README explains the concepts in prose, but the repository is meant to be clicked through. If you are a Python developer new to Rust extensions, read the code in this order:

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

## Start here: Basic Python list API

The first API deliberately looks like normal Python. This separates two questions: "does the package work?" and "why is one implementation faster?"

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

All list functions validate equal lengths, handle empty arrays, and return Python lists. That makes the API familiar, but it also means the Rust extension has to convert Python objects into Rust numbers and then convert the result back into Python objects.

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

## Faster buffer API

The list API is good for learning the call path, but it is not the data shape that fast numeric code wants. To understand the fast path, we need one low-level idea: a buffer.

First, what is a buffer?

A buffer is a view of raw memory that Python can expose to other code. NumPy arrays, `array.array`, `bytearray`, and `memoryview` are common examples. Instead of saying "here is a Python list of objects", a buffer says "here is a pointer to a block of bytes, here is the element type, here is the shape, and here is how to move from one element to the next".

If this idea is new, [An Introduction to the Python Buffer Protocol](https://jakevdp.github.io/blog/2014/05/05/introduction-to-the-python-buffer-protocol/) is the best next read. It explains why Python objects can expose raw memory to other code, which is exactly what makes `add_into(a, b, out)` avoid extra copies.

For this tutorial, the useful case is a NumPy `float32` array:

```text
NumPy array object
    |
    | exposes Python buffer protocol
    v
metadata: dtype=float32, ndim=1, contiguous=true, length=N
    |
    v
raw memory: [f32][f32][f32][f32]...
```

That is very different from a Python list:

```text
Python list
    |
    v
[ pointer ][ pointer ][ pointer ][ pointer ]...
      |         |         |         |
      v         v         v         v
 Python float Python float Python float Python float
```

A list is a container of Python objects. The list itself stores a contiguous array of pointers, but the actual Python float objects live somewhere else in memory. So the numbers are not laid out as one compact row of `float32` values in RAM.

That is a problem for SIMD because a vector load expects to grab several neighboring machine values with one instruction:

```text
what SIMD wants:

[f32][f32][f32][f32][f32][f32][f32][f32]
  ^ one vector load can read these neighboring values

what a Python list gives:

[ptr][ptr][ptr][ptr][ptr][ptr][ptr][ptr]
  |    |    |    |    |    |    |    |
  v    v    v    v    v    v    v    v
 float objects scattered around memory
```

Rust cannot treat those pointers as `f32` values. It must ask Python/PyO3 to read each Python object, convert it to a Rust `f32`, and place those converted values into a new contiguous `Vec<f32>`. That conversion is safe and convenient, but it costs time.

A numeric buffer is different: it is already a compact block of machine values. SIMD wants that second shape.

With that in mind, the list API copy is easier to read:

```text
Python list -> Rust Vec<f32> -> compute -> Python list
```

There are three expensive things in that short line:

```text
1. read millions of Python float objects
2. allocate and fill a Rust Vec<f32>
3. allocate and fill a new Python result list
```

For simple operations like addition, the arithmetic is extremely cheap. Adding two `f32` values is not the expensive part. Moving data between Python objects, Rust vectors, and Python result lists can dominate the whole runtime.

The buffer API changes what the benchmark is measuring. It moves from "how fast can Rust add?" plus "how expensive is Python object conversion?" to something much closer to "how fast can Rust process already-numeric memory?"

The faster API accepts Python buffers, such as NumPy `float32` arrays, and writes into a preallocated output array:

```python
import numpy as np
import nano_numpy_simd as nn

a = np.arange(1_000_000, dtype=np.float32)
b = np.arange(1_000_000, dtype=np.float32)
out = np.empty_like(a)

nn.add_into(a, b, out)
```

`add_into` returns `None`. The result is written into `out`.

The detail to notice is `out = np.empty_like(a)`. The output memory already exists before the timed operation. Rust only fills it:

```text
without preallocated output:
    allocate output memory
    compute values
    return new object

with preallocated output:
    reuse existing output memory
    compute values directly into it
```

For large arrays and simple math, allocation can cost as much as or more than the arithmetic. Preallocating `out` removes that cost from the hot loop.

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

## What FFI means

At this point we have two Python-facing APIs, but both eventually call Rust. The next question is: how can Python call Rust at all?

FFI means Foreign Function Interface. It is the mechanism that lets code written in one language call code written in another language.

For a deeper explanation of FFI, this video is a useful companion: [FFI explanation](https://www.youtube.com/watch?v=XJC5WB2Bwrc).

Python cannot directly call an ordinary Rust function because Rust has its own calling conventions, type system, ownership rules, name mangling, and memory model. PyO3 creates the CPython-compatible wrapper.

The final thing Python imports is a native shared library. On Linux this is usually a `.so` file. On macOS the extension also behaves like a shared library, even though the filename commonly ends with something like `.so` for Python extension compatibility. You can think of it as:

```text
Rust source code
      |
      | cargo and maturin compile it
      v
native shared library
      |
      | example name:
      | nano_numpy_simd/_native.cpython-311-darwin.so
      v
Python import system can load it
```

What is inside that `.so` file?

It is not Python source code. It is a compiled binary file containing machine code and metadata:

```text
nano_numpy_simd/_native.cpython-311-darwin.so

contains:
  - compiled Rust machine code
  - functions generated by PyO3
  - a CPython module initialization symbol
  - references to symbols the dynamic loader must resolve
  - metadata needed by the operating system loader
```

The filename carries useful information:

```text
_native        -> the extension module name
cpython-311    -> built for CPython 3.11
darwin         -> built for macOS
.so            -> Python extension shared library suffix
```

You can inspect the file yourself. First ask Python where the native extension lives:

```python
from nano_numpy_simd import _native
print(_native.__file__)
```

That prints a path like:

```text
/path/to/site-packages/nano_numpy_simd/_native.cpython-311-darwin.so
```

From the shell, save that path and ask the operating system what kind of file it is:

```bash
EXT_PATH="$(uv run python -c 'from nano_numpy_simd import _native; print(_native.__file__)')"
file "$EXT_PATH"
```

On macOS, the output may look like:

```text
nano_numpy_simd/_native.cpython-311-darwin.so: Mach-O 64-bit dynamically linked shared library arm64
```

That line says:

- `Mach-O`: macOS binary format. On Linux you would usually see `ELF`.
- `64-bit`: built for a 64-bit process.
- `dynamically linked shared library`: loaded by another program at runtime, here the Python interpreter.
- `arm64`: built for Apple Silicon / aarch64.

For this tutorial, the most useful question is not "what does every assembly instruction do?" The useful question is: what does Python see after it imports the compiled module?

Ask Python to list the functions exposed by the native module:

```python
from nano_numpy_simd import _native

names = [name for name in dir(_native) if not name.startswith("__")]
print(names)

for name in names:
    obj = getattr(_native, name)
    print(f"{name}: {obj}")
    print(f"  text_signature={getattr(obj, '__text_signature__', None)!r}")
```

Output:

```text
['add', 'add_into', 'add_rust', 'div', 'div_into', 'div_rust', 'mul', 'mul_into', 'mul_rust', 'sub', 'sub_into', 'sub_rust']
add: <built-in function add>
  text_signature='(a, b)'
add_into: <built-in function add_into>
  text_signature='(a, b, out)'
add_rust: <built-in function add_rust>
  text_signature='(a, b)'
div: <built-in function div>
  text_signature='(a, b)'
div_into: <built-in function div_into>
  text_signature='(a, b, out)'
div_rust: <built-in function div_rust>
  text_signature='(a, b)'
mul: <built-in function mul>
  text_signature='(a, b)'
mul_into: <built-in function mul_into>
  text_signature='(a, b, out)'
mul_rust: <built-in function mul_rust>
  text_signature='(a, b)'
sub: <built-in function sub>
  text_signature='(a, b)'
sub_into: <built-in function sub_into>
  text_signature='(a, b, out)'
sub_rust: <built-in function sub_rust>
  text_signature='(a, b)'
```

The phrase `<built-in function add>` means Python sees a native function implemented by an extension module, not a Python function defined with `def add(...)`.

This is the connection:

```text
Rust #[pyfunction] in src/lib.rs
        |
        | PyO3 registers it during module initialization
        v
Python sees <built-in function add>
        |
        | Python can call it like any other function
        v
_native.add([1.0], [2.0])
```

The low-level symbol to know is the module initializer.

On macOS, you can list exported symbols with:

```bash
nm -gU "$EXT_PATH"
```

On Linux, use:

```bash
nm -D "$EXT_PATH"
```

Important output:

```text
000000000000633c T _PyInit__native
```

`_PyInit__native` is the entry point CPython looks for when it imports `nano_numpy_simd._native`. The name matches the module name `_native`.
You do not normally call `_PyInit__native` yourself. CPython calls it during import. That initializer creates the Python module object and registers the Rust functions as Python-callable methods.

Why does this work? CPython knows how to load extension modules. When Python imports `nano_numpy_simd._native`, it asks the operating system loader to load the shared library into the Python process. That shared library contains a specially named module initialization function generated by PyO3. CPython calls that initializer, and the initializer registers functions such as `add`, `add_rust`, and `add_into`.

So the flow is not "Python interprets Rust". The flow is "Rust is compiled into machine code, packaged as a Python extension module, loaded into the Python process, and called through CPython's extension API".

You can even import the native module directly, although normal users should import `nano_numpy_simd`:

```python
from nano_numpy_simd import _native

print(_native.add([1.0, 2.0], [10.0, 20.0]))
# [11.0, 22.0]

print(_native.add_rust([1.0, 2.0], [10.0, 20.0]))
# [11.0, 22.0]
```

The public package re-exports those native functions in [`nano_numpy_simd/__init__.py`](nano_numpy_simd/__init__.py), so this is the recommended user-facing version:

```python
import nano_numpy_simd as nn

print(nn.add([1.0, 2.0], [10.0, 20.0]))
# [11.0, 22.0]
```

```text
Python code
    |
    | import nano_numpy_simd
    v
CPython import machinery
    |
    | loads compiled .so extension module
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

### Ownership at the boundary

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

## Memory layout

FFI explains how the call crosses from Python into Rust. Memory layout explains what Rust receives after that crossing. This matters because the CPU can only load neighboring values efficiently if those values are actually neighbors in memory.

For the full NumPy side of this topic, keep the [NumPy ndarray reference](https://numpy.org/doc/2.4/reference/arrays.ndarray.html) nearby. It is the official anchor for `dtype`, shape, strides, contiguity, item size, and why a NumPy `float32` array is so different from a Python list.

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

This shape also helps the CPU cache. Main memory is much slower than the CPU, so CPUs keep recently used memory in small, fast caches. When code reads `a[0]`, the CPU usually loads a whole cache line, not just one `f32`. A cache line is commonly 64 bytes, enough for 16 neighboring `float32` values.

```text
one cache line, 64 bytes

[a[0]][a[1]][a[2]][a[3]] ... [a[15]]
  ^ read one value
  |
  +-- nearby values are pulled into cache too
```

If the next loop iterations read `a[1]`, `a[2]`, and `a[3]`, those values may already be in cache. That is a cache hit: the CPU finds the data in its fast cache instead of waiting for main memory. Contiguous numeric arrays make cache hits more likely because the loop walks through neighboring addresses. Scattered Python objects make that harder because following each pointer can jump to a different place in memory.

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

## Scalar loop

Once Rust has slices pointing at contiguous `f32` values, the simplest correct implementation is a scalar loop. Scalar means "one value at a time".

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

The difference is that Rust is working with raw `f32` values in contiguous memory, not Python float objects. Python list iteration has interpreter overhead, dynamic objects, and reference counting. Rust's loop is compiled to machine code.

One caveat: in release mode, LLVM may auto-vectorize simple loops. So "naive Rust" means the source code is naive. It does not guarantee that the final assembly contains no vector instructions.

## SIMD in one picture

The scalar loop is the reference: it is easy to understand and works everywhere. SIMD keeps the same mathematical result but changes the width of each step.

SIMD means Single Instruction, Multiple Data. One instruction operates on several lanes.

Two words matter here:

- A register is a tiny, extremely fast storage location inside the CPU. CPU instructions usually operate on values in registers, not directly on Python objects.
- An instruction is one low-level operation the CPU can execute, such as "load memory", "add packed floats", or "store memory".

If you want to understand registers and how a CPU runs programs from the ground up, watch [Crafting a CPU to Run Programs](https://www.youtube.com/watch?v=GYlNoAMBY6o&t=11s) from Core Dumped.

Scalar code uses ordinary registers for one value at a time:

```text
register r0: a0
register r1: b0
instruction: add r0, r1
result: a0 + b0
```

SIMD uses wider vector registers. One vector register contains multiple lanes:

```text
vector register A: [a0, a1, a2, a3]
vector register B: [b0, b1, b2, b3]
instruction: add packed float32 lanes
result:            [a0+b0, a1+b1, a2+b2, a3+b3]
```

For Python developers, a helpful mental model is: SIMD is like doing a small fixed-size list operation inside one CPU instruction. The real hardware is more complex, but that intuition explains why contiguous numeric memory matters.

If you want a gentle next step, [Vectorization part 1. Intro — easyperf](https://easyperf.net/blog/2017/10/24/Vectorization_part1) shows scalar and vectorized loops side by side. The rest of the series is worth reading in this order for this project: vectorization warmup, compiler vectorization reports, vectorization width, multiversioning by data dependency, multiversioning by trip counts, and tips for writing vectorizable code.

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

The number of lanes processed by one SIMD instruction is the vectorization width. You can think of it as the fixed batch size of the CPU instruction.

The width depends on two things: the size of the vector register and the size of each element:

```text
vectorization width = SIMD register size / element size
```

For example, a 256-bit AVX2 register can hold eight `f32` values because each `f32` is 32 bits:

```text
256 bits / 32 bits = 8 lanes
```

The same register holds only four `f64` values:

```text
256 bits / 64 bits = 4 lanes
```

This is why SIMD code usually moves through an array in fixed-size chunks, then uses a scalar tail loop for the remaining values when the array length is not a multiple of the vectorization width.

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

## Runtime dispatch

SIMD sounds like one thing, but different CPUs speak different SIMD "dialects". x86_64 and aarch64 do not use the same instruction names or registers.

The Rust code must not execute instructions unsupported by the current CPU. That can crash the process with an illegal instruction.

Each CPU architecture has its own instruction sets:

- x86_64 CPUs may support SSE, AVX, AVX2, AVX-512, and more.
- aarch64 CPUs use NEON as the common SIMD instruction set.
- other targets may not have an implementation in this tutorial, so they use scalar Rust.

Dispatch means: choose the correct set of instructions for the current architecture and CPU features.

```text
same Python call: nn.add(a, b)
          |
          v
   what architecture?
          |
    +-----+------+----------------+
    |            |                |
  x86_64      aarch64          other
    |            |                |
    v            v                v
check CPU      use NEON         scalar
features
    |
    +--> AVX2 if supported
    +--> SSE if AVX2 is not supported
    +--> scalar if no SIMD path is selected
```

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

The dispatch layer is small, but it does real work. The Python function `nn.add(a, b)` does not know what CPU it is running on. The Rust dispatcher checks the target and available features, then calls the best supported implementation.

For a deeper look at why simple-looking SIMD problems become subtle, read [Understanding SIMD: Infinite Complexity of Trivial Problems](https://www.modular.com/blog/understanding-simd-infinite-complexity-of-trivial-problems). It pairs well with runtime dispatch, platform differences, and the caveats near the end of this README.

## AVX2 loop shape

After dispatch chooses AVX2 on x86_64, the loop can use 256-bit YMM registers. This is the widest x86_64 path implemented in the tutorial.

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

If you want to see those intrinsics explained visually, [SIMD and vectorization using AVX intrinsic functions](https://www.youtube.com/watch?v=AT5nuQQO96o) is a good companion while reading `src/simd_x86.rs`.

SSE in the same file uses 128-bit XMM registers and processes 4 `f32` values per iteration. That gives the tutorial a smaller x86_64 fallback before scalar code.

## NEON loop shape

On aarch64, the tutorial uses NEON. NEON uses 128-bit vector registers for this example, so it processes fewer `f32` lanes per instruction than AVX2, but the idea is the same: load a chunk, compute all lanes, store the chunk.

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

## Inspecting generated assembly

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

## Tail handling

SIMD loops work in fixed-size chunks, but real arrays can have any length. That is why every SIMD implementation needs tail handling.

Not every array length is divisible by 8 or 4. The SIMD loop handles full vector chunks, then scalar code handles the rest.

```text
len = 17, AVX2 width = 8

indexes:  0 1 2 3 4 5 6 7 | 8 9 10 11 12 13 14 15 | 16
          <---- AVX2 ----> | <------- AVX2 -------> | scalar tail
```

The tests include lengths `0, 1, 3, 4, 5, 7, 8, 9, 15, 16, 17` to exercise both vector chunks and tails.

## What happens inside the CPU?

We now have enough pieces to describe the CPU work at a beginner level. Python has called Rust, Rust has selected an implementation, and the SIMD loop is running over contiguous memory.

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

## Why the list benchmark is only a little faster

Now return to the benchmark from the top of the README. The list API is not slow because Rust cannot add numbers quickly. It is limited because the input and output are Python object containers.

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

Returning to the same Apple Silicon run from the top of the README:

```text
Size: 5,000,000 float32 values
Implementation     Time           Speedup vs Python
Pure Python list      180.690 ms       1.00x
Rust list API         126.435 ms       1.43x
Rust buffer into        0.785 ms     230.13x
NumPy allocated         0.776 ms     232.82x
NumPy out=              0.770 ms     234.65x
```

These numbers are examples from one machine. Run the benchmark locally.

## Running the benchmarks

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

## Safety notes

The fast path is low-level, so the code has to be explicit about safety. Rust normally protects you from invalid memory access, but SIMD intrinsics and Python buffers require carefully checked `unsafe` blocks.

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

## What would be next to become a real NumPy

This project is intentionally small. It teaches the path from Python lists to Rust buffers and SIMD kernels. A real NumPy-like library needs several larger components around that core.

### Array object

The first missing piece is a real array type. It would own or borrow memory and store the metadata needed to interpret that memory.

A pointer or buffer owner tells the array where the actual bytes live and who is responsible for keeping that memory alive. Without this, the array cannot safely know whether the memory is still valid.

The shape tells the array how many dimensions it has and how large each dimension is. For example, shape `(2, 3)` means two rows and three columns.

The strides tell the array how many bytes to jump in memory when moving by one step along each dimension. This is what makes slices, transposes, and non-contiguous views possible.

The dtype tells the array how to interpret the bytes. The same raw memory could be read as `float32`, `int64`, `bool`, or another type depending on the dtype metadata.

The length tells the array how many logical elements it contains. This is useful for validation, iteration, bounds checks, and deciding how much work a kernel has to do.

The byte offset lets a view start in the middle of another array's memory without copying. A slice can point to the same storage but begin at a later byte.

The read-only or writable flag prevents unsafe writes into memory that should not be modified. This matters when memory comes from another Python object, a shared view, or an external buffer.

With this, the library can represent more than flat contiguous `f32` arrays. It can represent views, slices, transposes, reshapes, and arrays that share the same memory.

### Dtype system

NumPy supports many data types. A real version would need a dtype system that explains how raw bytes should be interpreted and how different types interact.

Element size tells the array how many bytes each value occupies. A `float32` uses `4` bytes, while a `float64` uses `8` bytes, so the same memory buffer length can contain different numbers of logical elements depending on the dtype.

Alignment describes where values should start in memory for efficient and valid access. Some CPUs can read unaligned values, but aligned memory is often faster and safer for SIMD kernels.

Signed and unsigned integers need different interpretation rules. The same byte pattern can mean a negative number for a signed integer or a large positive number for an unsigned integer.

Floats need their own handling because they follow floating-point rules. They can represent fractional values, infinities, and `NaN`, and operations on them may have rounding behavior.

Booleans are another dtype family. They are often stored compactly and are commonly produced by comparisons such as `array > 0`.

Type promotion rules decide the result dtype when different input dtypes meet. For example, adding an integer array to a floating-point array usually produces a floating-point result.

Casting rules decide when one dtype can be converted to another. Some casts are safe, such as `int32` to `int64`, while others can lose information, such as `float64` to `int32`.

The current code only handles `f32`. Supporting `f64`, integers, and booleans would require either generic kernels, generated kernels, or carefully organized per-dtype implementations.

### Shape and stride engine

Shapes and strides are what make multidimensional arrays work without copying data. They let one flat memory buffer behave like a matrix, a tensor, a slice, or a transposed view.

Indexing uses the shape and strides to turn a logical position, such as row `1` and column `2`, into a byte address inside the flat buffer.

Slicing creates a smaller view of the same memory. Instead of copying values, the array can change its shape, strides, and byte offset to describe the selected region.

Reshaping changes how the same elements are grouped into dimensions. When the memory layout allows it, reshape can be metadata-only. When the layout is too irregular, it may require a copy.

Transposing swaps dimensions by changing the order of the shape and strides. This is why a transpose can often be created instantly without moving any data.

Contiguous layouts store neighboring logical elements next to each other in memory. Non-contiguous layouts appear after operations like slicing with a step or transposing an array.

Row-major and column-major views describe which dimension moves fastest in memory. C and NumPy usually use row-major order by default, while Fortran and many linear algebra libraries often use column-major order.

For a simple row-major `2 x 3` array, the data may be stored as one flat buffer:

```text
Logical array:

  [[1, 2, 3],
   [4, 5, 6]]

Flat memory:

  index:  0   1   2   3   4   5
  value:  1   2   3   4   5   6

Metadata:

  shape   = (2, 3)
  strides = (12, 4)   # for float32: 3 values per row * 4 bytes, then 1 value * 4 bytes
```

The shape says this is a matrix with `2` rows and `3` columns. The strides say how to move through the flat memory. To move down one row, jump `12` bytes. To move right one column, jump `4` bytes.

A transpose can often be represented by changing only the metadata:

```text
Original:

  shape   = (2, 3)
  strides = (12, 4)

Transposed view:

  shape   = (3, 2)
  strides = (4, 12)
```

The underlying bytes do not move. The array just reads them in a different order.

Many operations should work on strided views directly. When that is too slow or too complex, the library can copy into a contiguous temporary buffer.

### Broadcasting

Broadcasting lets operations combine arrays with different but compatible shapes. For example, adding a vector to every row of a matrix requires shape analysis before the kernel runs.

The first step is to compare input shapes from the right. The rightmost dimensions describe the innermost elements, so those are matched first.

Dimensions of size `1` can be expanded virtually. This means the value is reused across a larger dimension without physically copying it.

After checking compatibility, the engine computes the output shape. For example, combining shape `(2, 3)` with shape `(3,)` produces shape `(2, 3)`.

Broadcasted dimensions use virtual strides. If a value is reused along an expanded dimension, the stride for that dimension can behave like `0`, meaning the kernel keeps reading the same value while the other array moves forward.

If two dimensions are incompatible, the operation should fail before the kernel runs. For example, shapes `(2, 3)` and `(4,)` cannot be broadcast together because `3` and `4` do not match and neither one is `1`.

### Universal functions

NumPy's elementwise operations are implemented as ufuncs. A ufunc is a reusable operation object that knows how to apply the same operation across arrays.

The simplest ufuncs are arithmetic operations such as `add`, `subtract`, `multiply`, and `divide`. They usually read one element from each input array and write one element to the output array.

Comparisons are also ufuncs. Operations such as `<`, `>`, and `==` still walk over the input arrays element by element, but they produce boolean output instead of numeric output.

Math functions such as `sin`, `exp`, and `sqrt` are ufuncs too. They often need specialized implementations because they are more complex than basic arithmetic and may use approximations, lookup strategies, or hardware instructions depending on the dtype and CPU.

The important part is that all these operations can share one execution model. Each ufunc chooses the right inner loop based on dtype, shape, strides, broadcasting, and CPU features.

### Memory management

A real array library needs clear ownership rules. It must know whether memory comes from Python, Rust, another array view, or an external buffer.

The first important piece is reference-counted storage. Multiple arrays can point to the same underlying memory, so the storage must stay alive until the last array or view stops using it.

Once storage can be shared, the library needs safe views into that shared storage. A slice, reshape, or transpose should be able to reuse the original bytes without copying them, but it must still respect bounds, dtype, shape, and strides.

Mutation makes this more delicate. If two arrays share the same memory, writing through one view can change what the other view sees. Some libraries allow that directly, while others use copy-on-write decisions to copy the data before mutation when sharing would be surprising or unsafe.

Memory alignment is another performance detail. SIMD instructions often work best when data starts at addresses aligned to specific byte boundaries, so the allocator and array metadata should preserve useful alignment guarantees when possible.

Temporary allocation reuse helps avoid repeated memory allocation inside chained operations. For example, an expression may need short-lived intermediate buffers, and reusing those buffers can reduce overhead.

Finally, safe interoperability with the Python buffer protocol lets the array exchange memory with Python objects such as `array.array`, `memoryview`, and NumPy arrays. This is necessary for zero-copy interoperability, but it requires careful lifetime and mutability checks.

### Kernel dispatch

The current project already has a small dispatch layer. A larger version would expand it into a full kernel selection system:

A scalar fallback is the simplest implementation. It processes one value at a time and works on every CPU, so it is the safe baseline when no specialized instruction set is available.

Once that baseline exists, the next question is what the current CPU can do faster.

SIMD kernels per architecture use CPU-specific vector instructions. An x86_64 machine may use SSE, AVX, AVX2, or AVX-512, while an aarch64 machine may use NEON. These kernels are necessary because different processors expose different vector registers and instructions.

After choosing a CPU-specific path, the dispatch layer also has to look at the memory layout.

Contiguous fast paths handle arrays whose elements are stored next to each other in memory. This is the easiest layout for the CPU to read efficiently, so it is usually where the fastest loops and SIMD kernels live.

Not every array is contiguous, though. Views can point to the same memory while walking through it differently.

Strided loops handle views where neighboring logical elements are not necessarily neighboring memory addresses. They are necessary for slices, transposes, and other views that should avoid copying data.

The best choice also depends on how much work there is.

Small-array paths avoid spending more time on setup than on computation. For tiny inputs, checking CPU features, preparing threads, or entering a complex SIMD loop can cost more than a simple scalar loop.

For large inputs, the trade-off changes again.

Multithreaded paths split large arrays across CPU cores. They are useful when the array is big enough that the extra coordination cost is smaller than the time saved by doing work in parallel.

The dispatch layer should hide CPU-specific details from the Python API.

### Reductions and aggregations

Operations like `sum`, `mean`, `min`, `max`, and `argmax` need different kernels from elementwise operations. They also need axis handling.

Elementwise kernels usually read one or more input elements and write one output element for each position:

```text
add:

  output[i] = left[i] + right[i]
```

Reduction kernels combine many input values into fewer output values:

```text
sum over all elements:

  output = input[0] + input[1] + input[2] + ...
```

That changes the kernel design. A reduction needs an accumulator, an initial value, a loop order, and rules for the output dtype. For example, summing `int8` values into an `int8` accumulator can overflow quickly, so real array libraries often use wider accumulation types.

Reductions can use special vectorized instructions, but they are less direct than elementwise operations. For elementwise `add`, SIMD can load several values from the left array, several values from the right array, add them, and store several results. For `sum`, SIMD usually keeps several partial sums inside vector registers, then combines those partial sums at the end into one final value.

```text
SIMD-style sum reduction idea:

  assume one 128-bit SIMD register
  assume float32 values

  128 bits / 32 bits per float32 = 4 lanes per register

  input:        [1, 2, 3, 4, 5, 6, 7, 8]

  load 1:       [1, 2, 3, 4]
  load 2:       [5, 6, 7, 8]

  vector add:   [1+5, 2+6, 3+7, 4+8]
  partials:     [6, 8, 10, 12]

  final sum:    6 + 8 + 10 + 12 = 36
```

This is why reductions often have separate kernels for contiguous arrays, strided arrays, small arrays, different dtypes, and different CPU features. The fastest path depends on memory layout and on whether the CPU supports instructions such as SSE, AVX, AVX2, AVX-512, or NEON.

Axis handling decides which dimension is reduced. For a `2 x 3` matrix:

```text
input:

  [[1, 2, 3],
   [4, 5, 6]]

sum over axis 0:

  [5, 7, 9]     # columns are reduced

sum over axis 1:

  [6, 15]       # rows are reduced
```


### Linear algebra

NumPy relies on highly optimized BLAS and LAPACK libraries for many linear algebra operations. A real project should usually call existing libraries instead of reimplementing everything.

BLAS means Basic Linear Algebra Subprograms. It is a standard set of low-level routines for operations such as vector addition, dot products, matrix-vector multiplication, and matrix-matrix multiplication. These operations are simple to describe but hard to make extremely fast on real hardware.

LAPACK means Linear Algebra Package. It builds on BLAS and provides higher-level algorithms such as matrix decompositions, eigenvalue routines, and solving systems of linear equations.

The important idea is that BLAS and LAPACK are not one specific library. They are standard APIs. Different implementations provide those APIs with heavy optimization for different CPUs.

Historically, BLAS and LAPACK were written in Fortran. Many modern implementations still expose Fortran-compatible interfaces, but their internals may use C, C++, Fortran, assembly, and CPU-specific intrinsic code. Under the hood, the fastest parts are often hand-tuned kernels for each processor family.

Common implementations include:

- OpenBLAS: mostly C and assembly, with kernels for many CPU architectures
- BLIS: C framework with optimized low-level kernels
- Intel MKL: proprietary Intel library with highly optimized CPU kernels
- Accelerate: Apple's macOS framework, optimized for Apple platforms
- Reference BLAS/LAPACK: mostly Fortran, useful as a correctness baseline but usually not the fastest option

## Closing words
By now, vectorization should not feel like magic anymore. It felt that way to me at first, but understanding it demystifies the trick and makes the whole thing even more wonderful. If you can see that too, I am happy, because it means you read the whole tutorial.
