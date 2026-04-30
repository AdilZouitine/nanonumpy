# FFI: how Python calls Rust

### 1. What is FFI?

FFI means Foreign Function Interface. It is the mechanism that allows code written in one language to call code written in another language.

### 2. Why Python cannot directly call normal Rust functions

A normal Rust function has Rust-specific calling conventions, type system rules, name mangling, ownership rules, and memory safety assumptions. Python cannot directly call that function unless it is exposed through a compatible interface.

### 3. The three common ways to call native code from Python

CPython extension modules load native code into the Python process. `ctypes` can call functions from dynamic/shared libraries using C-compatible types. CFFI lets Python interact with C-compatible APIs using C-like declarations. PyO3 builds a CPython extension module so Python can import Rust code as if it were a normal Python module.

### 4. What PyO3 does in this project

PyO3 creates the glue code between CPython and Rust. `#[pyfunction]` exposes a Rust function as a Python-callable function. `#[pymodule]` creates the Python module. PyO3 converts Python objects into Rust values where possible, converts Rust return values back into Python objects, and lets Rust create Python exceptions from Rust errors.

### 5. What maturin does

maturin builds the Rust crate as a Python extension module. It produces a shared library with a Python-compatible name. `maturin develop` builds and installs the module into the current virtual environment. After that, Python can run `import nano_numpy_simd`.

### 6. FFI call path

```text
Python code
  |
  | import nano_numpy_simd
  v
CPython import machinery
  |
  | loads compiled shared object
  v
PyO3 module initializer
  |
  | registers #[pyfunction] functions
  v
Python calls nn.add(a, b)
  |
  | PyO3 extracts Python list into Rust Vec<f32>
  v
Rust function receives &[f32]
  |
  | dispatches to scalar / SIMD
  v
Rust returns Vec<f32>
  |
  | PyO3 converts Vec<f32> back to Python list
  v
Python receives result
```

### 7. Ownership and memory

Python owns the original list. PyO3 converts or copies values into Rust-owned memory. Rust owns the `Vec<f32>` during computation. The result `Vec<f32>` is converted back to a Python object. This copy is simple and safe for the tutorial, but it creates overhead. Future work could use NumPy arrays or the Python buffer protocol to reduce copying.

### 8. Why FFI overhead matters

Calling Rust from Python is not free. There is overhead for crossing the Python/Rust boundary, converting Python lists to Rust vectors, allocating output vectors, and converting results back to Python. For tiny arrays, this overhead may be larger than the SIMD speedup.

### 9. Optional low-level FFI example

```rust
#[no_mangle]
pub extern "C" fn add_f32_ptrs(
    a: *const f32,
    b: *const f32,
    out: *mut f32,
    len: usize,
) {
    // pointer-based implementation
}
```

This kind of function could be called with `ctypes` or CFFI, but it is unsafe because Python must pass valid pointers, valid lengths, and writable output memory. The actual project uses PyO3 because it is safer and more ergonomic for Python users.
