//! High level bindings to the cuBLAS CUDA library for high performance GPU
//! BLAS (Basic Linear Algebra Subroutines).
//!
//! # Indexing
//!
//! **blastoff uses 1-based indexing, reflecting cuBLAS' behavior. This means
//! you will likely need to do some math to any returned indices. For example,
//! [`amin`](crate::context::CublasContext::amin) returns a 1-based index.**

#![allow(clippy::too_many_arguments)]
#![cfg_attr(docsrs, feature(doc_cfg))]

pub use cublas_sys as sys;
use num_complex::{Complex32, Complex64};

pub use context::*;

mod context;
pub mod error;
mod level1;
mod level3;
pub mod raw;

/// A possible datatype for a generic matrix mul operation. This is just [`BlasDatatype`] except optionally
/// containing `f16` with the `half` feature.
pub trait GemmDatatype: private::Sealed + cust::memory::DeviceCopy {}

#[cfg(feature = "half")]
impl private::Sealed for half::f16 {}
#[cfg_attr(docsrs, doc(cfg(feature = "half")))]
#[cfg(feature = "half")]
impl GemmDatatype for half::f16 {}
impl GemmDatatype for f32 {}
impl GemmDatatype for f64 {}
impl GemmDatatype for Complex32 {}
impl GemmDatatype for Complex64 {}

pub trait BlasDatatype: private::Sealed + cust::memory::DeviceCopy {
    /// The corresponding float type. For complex numbers this means their backing
    /// precision, and for floats it is just themselves.
    type FloatTy: Float;
    fn to_raw(&self) -> sys::v2::cudaDataType;
}

impl BlasDatatype for f32 {
    type FloatTy = f32;
    fn to_raw(&self) -> sys::v2::cudaDataType {
        sys::v2::cudaDataType::CUDA_R_32F
    }
}

impl BlasDatatype for f64 {
    type FloatTy = f64;
    fn to_raw(&self) -> sys::v2::cudaDataType {
        sys::v2::cudaDataType::CUDA_R_64F
    }
}

impl BlasDatatype for Complex32 {
    type FloatTy = f32;
    fn to_raw(&self) -> sys::v2::cudaDataType {
        sys::v2::cudaDataType::CUDA_C_32F
    }
}

impl BlasDatatype for Complex64 {
    type FloatTy = f64;
    fn to_raw(&self) -> sys::v2::cudaDataType {
        sys::v2::cudaDataType::CUDA_C_64F
    }
}

/// Trait describing either 32 or 64 bit complex numbers.
pub trait Complex: private::Sealed + BlasDatatype {}
impl Complex for Complex32 {}
impl Complex for Complex64 {}

/// Trait describing either 32 or 64 bit floats.
pub trait Float: private::Sealed + BlasDatatype {}
impl Float for f32 {}
impl Float for f64 {}

pub(crate) mod private {
    use num_complex::{Complex32, Complex64};

    pub trait Sealed {}

    impl Sealed for f32 {}
    impl Sealed for f64 {}
    impl Sealed for Complex32 {}
    impl Sealed for Complex64 {}
}

/// An optional operation to apply to a matrix before a matrix operation. This includes
/// no operation, transpose, or conjugate transpose.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MatrixOp {
    /// No operation, leave the matrix as is. This is the default.
    #[default]
    None,
    /// Transpose the matrix in place.
    Transpose,
    /// Conjugate transpose the matrix in place.
    ConjugateTranspose,
}

impl MatrixOp {
    /// Returns the corresponding `cublasOperation_t` for this operation.
    pub fn to_raw(self) -> sys::v2::cublasOperation_t {
        match self {
            MatrixOp::None => sys::v2::cublasOperation_t::CUBLAS_OP_N,
            MatrixOp::Transpose => sys::v2::cublasOperation_t::CUBLAS_OP_T,
            MatrixOp::ConjugateTranspose => sys::v2::cublasOperation_t::CUBLAS_OP_C,
        }
    }
}
