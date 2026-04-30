//! Educational notes about intrinsics and conceptual assembly.
//!
//! The SIMD intrinsics live in `simd_x86.rs` and `simd_arm.rs`. This file points
//! readers toward the idea that intrinsics are lowered by the compiler.

/// Short explanation used by the tutorial docs.
pub const SUMMARY: &str =
    "SIMD intrinsics describe vector operations that the compiler lowers to machine instructions.";
