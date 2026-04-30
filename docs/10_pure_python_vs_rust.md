# Pure Python vs Rust

Pure Python lists store Python objects, not raw `f32` values. Each element operation involves interpreter overhead, dynamic dispatch, reference counting, and boxed number objects.

Rust receives contiguous `Vec<f32>` values after PyO3 conversion. A compiled Rust loop can load raw floats, apply a machine operation, and write raw floats without Python per-element overhead.

That does not mean Rust calls are free. The Python lists still need to be converted into Rust vectors, and the result needs to be converted back. For small arrays, this boundary cost can hide the speedup from compiled code.
