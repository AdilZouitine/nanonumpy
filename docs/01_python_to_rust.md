# Python to Rust

Python imports `nano_numpy_simd`, which loads a compiled CPython extension built by maturin and PyO3. The public Python name `nn.add` is backed by a Rust function marked with `#[pyfunction]` in `src/lib.rs`.

When you call `nn.add([1.0], [2.0])`, PyO3 converts Python lists into Rust `Vec<f32>` values. The Rust code validates the lengths, dispatches to scalar or SIMD code, returns a `Vec<f32>`, and PyO3 converts that result back into a Python list.

This boundary is called FFI: a Foreign Function Interface. It is powerful, but it is not free. Copying Python lists into Rust vectors can dominate the runtime for tiny arrays.
