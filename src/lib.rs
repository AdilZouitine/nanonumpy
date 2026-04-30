//! PyO3 extension module for nano-numpy-simd.
//!
//! This is the main Python/Rust boundary. Python calls functions registered in
//! this file. PyO3 converts Python objects into Rust values, the Rust operation
//! layer computes the result, and PyO3 converts the return value back to Python.

pub mod dispatch;
pub mod naive_rust;
pub mod ops;
pub mod scalar;

#[cfg(target_arch = "aarch64")]
pub mod simd_arm;
#[cfg(target_arch = "x86_64")]
pub mod simd_x86;

pub mod asm_explain;
pub mod ffi_explain;

use ops::Op;
use pyo3::buffer::PyBuffer;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::slice;

fn rust_error_to_py(err: String) -> PyErr {
    // Internal Rust helpers return `Result<_, String>` so they stay independent
    // from PyO3. At the boundary, strings become Python `ValueError`s.
    PyValueError::new_err(err)
}

#[pyfunction]
fn add_rust(a: Vec<f32>, b: Vec<f32>) -> PyResult<Vec<f32>> {
    // PyO3 has already converted Python sequences into owned Rust vectors.
    // This path is "naive Rust": a compiled loop, no explicit SIMD intrinsics.
    ops::elementwise_naive_rust(&a, &b, Op::Add).map_err(rust_error_to_py)
}

#[pyfunction]
fn sub_rust(a: Vec<f32>, b: Vec<f32>) -> PyResult<Vec<f32>> {
    ops::elementwise_naive_rust(&a, &b, Op::Sub).map_err(rust_error_to_py)
}

#[pyfunction]
fn mul_rust(a: Vec<f32>, b: Vec<f32>) -> PyResult<Vec<f32>> {
    ops::elementwise_naive_rust(&a, &b, Op::Mul).map_err(rust_error_to_py)
}

#[pyfunction]
fn div_rust(a: Vec<f32>, b: Vec<f32>) -> PyResult<Vec<f32>> {
    ops::elementwise_naive_rust(&a, &b, Op::Div).map_err(rust_error_to_py)
}

#[pyfunction]
fn add(a: Vec<f32>, b: Vec<f32>) -> PyResult<Vec<f32>> {
    // Same list-style API as `add_rust`, but the computation goes through the
    // runtime dispatcher and may use AVX2, SSE, NEON, or scalar fallback.
    ops::elementwise_simd(&a, &b, Op::Add).map_err(rust_error_to_py)
}

#[pyfunction]
fn sub(a: Vec<f32>, b: Vec<f32>) -> PyResult<Vec<f32>> {
    ops::elementwise_simd(&a, &b, Op::Sub).map_err(rust_error_to_py)
}

#[pyfunction]
fn mul(a: Vec<f32>, b: Vec<f32>) -> PyResult<Vec<f32>> {
    ops::elementwise_simd(&a, &b, Op::Mul).map_err(rust_error_to_py)
}

#[pyfunction]
fn div(a: Vec<f32>, b: Vec<f32>) -> PyResult<Vec<f32>> {
    ops::elementwise_simd(&a, &b, Op::Div).map_err(rust_error_to_py)
}

fn validate_buffer_inputs(
    a: &PyBuffer<f32>,
    b: &PyBuffer<f32>,
    out: &PyBuffer<f32>,
) -> PyResult<usize> {
    // The fast path is deliberately narrow: one-dimensional contiguous f32
    // buffers. That maps cleanly to Rust slices and SIMD loads.
    if a.dimensions() != 1 || b.dimensions() != 1 || out.dimensions() != 1 {
        return Err(PyValueError::new_err("buffers must be one-dimensional"));
    }

    let len = a.item_count();
    // No broadcasting in this tutorial. Every implementation computes
    // pairwise `a[i] op b[i]`.
    if b.item_count() != len || out.item_count() != len {
        return Err(PyValueError::new_err("arrays must have the same length"));
    }

    // C-contiguous means logical neighbors are physical neighbors in memory,
    // which is exactly what vector loads expect.
    if !a.is_c_contiguous() || !b.is_c_contiguous() || !out.is_c_contiguous() {
        return Err(PyValueError::new_err("buffers must be C-contiguous"));
    }

    // The `*_into` functions write directly into `out`, so a read-only NumPy
    // view or memoryview cannot be accepted.
    if out.readonly() {
        return Err(PyValueError::new_err("output buffer must be writable"));
    }

    // Rust requires unique mutable access. Rejecting overlap keeps `out` from
    // aliasing either input while we create `&mut [f32]`.
    if buffers_overlap(a, out) || buffers_overlap(b, out) {
        return Err(PyValueError::new_err(
            "output buffer must not overlap input buffers",
        ));
    }

    Ok(len)
}

fn buffers_overlap(left: &PyBuffer<f32>, right: &PyBuffer<f32>) -> bool {
    // Compare half-open byte ranges: [start, end). Two ranges overlap when
    // each starts before the other ends.
    let left_start = left.buf_ptr() as usize;
    let left_end = left_start + left.len_bytes();
    let right_start = right.buf_ptr() as usize;
    let right_end = right_start + right.len_bytes();

    left_start < right_end && right_start < left_end
}

fn elementwise_into_buffer(
    _py: Python<'_>,
    a: PyBuffer<f32>,
    b: PyBuffer<f32>,
    out: PyBuffer<f32>,
    op: Op,
) -> PyResult<()> {
    let len = validate_buffer_inputs(&a, &b, &out)?;

    // SAFETY: PyBuffer validated f32 format and alignment. The checks above ensure all
    // buffers are one-dimensional, C-contiguous, equal length, and that `out` is writable
    // and non-overlapping with both inputs while the GIL holds the buffer exports alive.
    let (a_slice, b_slice, out_slice) = unsafe {
        (
            // These slices borrow memory owned by Python objects. No element
            // conversion happens here; this is why the buffer API is fast.
            slice::from_raw_parts(a.buf_ptr().cast::<f32>(), len),
            slice::from_raw_parts(b.buf_ptr().cast::<f32>(), len),
            slice::from_raw_parts_mut(out.buf_ptr().cast::<f32>(), len),
        )
    };

    ops::validate_lengths(a_slice, b_slice).map_err(rust_error_to_py)?;
    // Reuse the exact same dispatcher as the list API. The difference is only
    // how inputs and outputs cross the Python/Rust boundary.
    dispatch::dispatch_elementwise(a_slice, b_slice, out_slice, op);
    Ok(())
}

#[pyfunction]
fn add_into(
    py: Python<'_>,
    a: PyBuffer<f32>,
    b: PyBuffer<f32>,
    out: PyBuffer<f32>,
) -> PyResult<()> {
    elementwise_into_buffer(py, a, b, out, Op::Add)
}

#[pyfunction]
fn sub_into(
    py: Python<'_>,
    a: PyBuffer<f32>,
    b: PyBuffer<f32>,
    out: PyBuffer<f32>,
) -> PyResult<()> {
    elementwise_into_buffer(py, a, b, out, Op::Sub)
}

#[pyfunction]
fn mul_into(
    py: Python<'_>,
    a: PyBuffer<f32>,
    b: PyBuffer<f32>,
    out: PyBuffer<f32>,
) -> PyResult<()> {
    elementwise_into_buffer(py, a, b, out, Op::Mul)
}

#[pyfunction]
fn div_into(
    py: Python<'_>,
    a: PyBuffer<f32>,
    b: PyBuffer<f32>,
    out: PyBuffer<f32>,
) -> PyResult<()> {
    elementwise_into_buffer(py, a, b, out, Op::Div)
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Register every Rust function that Python should be able to import from
    // `nano_numpy_simd._native`.
    m.add_function(wrap_pyfunction!(add_rust, m)?)?;
    m.add_function(wrap_pyfunction!(sub_rust, m)?)?;
    m.add_function(wrap_pyfunction!(mul_rust, m)?)?;
    m.add_function(wrap_pyfunction!(div_rust, m)?)?;
    m.add_function(wrap_pyfunction!(add, m)?)?;
    m.add_function(wrap_pyfunction!(sub, m)?)?;
    m.add_function(wrap_pyfunction!(mul, m)?)?;
    m.add_function(wrap_pyfunction!(div, m)?)?;
    m.add_function(wrap_pyfunction!(add_into, m)?)?;
    m.add_function(wrap_pyfunction!(sub_into, m)?)?;
    m.add_function(wrap_pyfunction!(mul_into, m)?)?;
    m.add_function(wrap_pyfunction!(div_into, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simd_api_validates_lengths() {
        let err = ops::elementwise_simd(&[1.0], &[1.0, 2.0], Op::Add).unwrap_err();

        assert_eq!(err, "arrays must have the same length");
    }

    #[test]
    fn naive_and_dispatch_match() {
        let a: Vec<f32> = (0..17).map(|x| x as f32).collect();
        let b: Vec<f32> = (0..17).map(|x| (x as f32) + 1.0).collect();

        let naive = ops::elementwise_naive_rust(&a, &b, Op::Sub).unwrap();
        let simd = ops::elementwise_simd(&a, &b, Op::Sub).unwrap();

        assert_eq!(naive, simd);
    }
}
