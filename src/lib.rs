//! PyO3 extension module for nano-numpy-simd.

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
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

fn rust_error_to_py(err: String) -> PyErr {
    PyValueError::new_err(err)
}

#[pyfunction]
fn add_rust(a: Vec<f32>, b: Vec<f32>) -> PyResult<Vec<f32>> {
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

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(add_rust, m)?)?;
    m.add_function(wrap_pyfunction!(sub_rust, m)?)?;
    m.add_function(wrap_pyfunction!(mul_rust, m)?)?;
    m.add_function(wrap_pyfunction!(div_rust, m)?)?;
    m.add_function(wrap_pyfunction!(add, m)?)?;
    m.add_function(wrap_pyfunction!(sub, m)?)?;
    m.add_function(wrap_pyfunction!(mul, m)?)?;
    m.add_function(wrap_pyfunction!(div, m)?)?;
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
