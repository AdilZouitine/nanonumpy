# Benchmarking

Benchmarks are noisy. CPU frequency, cache state, background processes, compiler flags, and thermal throttling can all change timings.

Python overhead matters in this project because the API starts with Python lists. PyO3 converts those lists into Rust `Vec<f32>` values and converts the result back into a Python list. For small arrays, that conversion can cost more than the arithmetic.

Large arrays usually show SIMD more clearly because the fixed boundary cost is spread over more work. Even then, NumPy may still be faster because it is mature, heavily optimized, and operates directly on contiguous array buffers.

Use `python python/examples/full_performance_comparison.py` for the Python-facing comparison and `cargo bench` for the Rust-only benchmark.
