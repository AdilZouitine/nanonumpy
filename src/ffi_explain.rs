//! Educational notes about the Python/Rust FFI boundary.
//!
//! The real bindings live in `lib.rs`. This file exists so readers can click a
//! source file dedicated to the FFI explanation from the README.

/// Short explanation used by the tutorial docs.
pub const SUMMARY: &str = "PyO3 exposes Rust functions as CPython extension-module functions.";
