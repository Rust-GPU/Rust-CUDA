//! Bindings to the CUDA Basic Linear Algebra Subprograms (cuBLAS) library.
#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

use crate::types::library::*;

pub use crate::runtime_sys::cudaStream_t;
pub use crate::types::complex::*;
pub use crate::types::library::cudaDataType;

include!(concat!(env!("OUT_DIR"), "/cublas_sys.rs"));

#[cfg(feature = "cublasLt")]
pub mod lt;

#[cfg(feature = "cublasXt")]
pub mod xt;
