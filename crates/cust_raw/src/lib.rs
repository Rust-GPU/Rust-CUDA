#[cfg(feature = "driver")]
#[allow(clippy::missing_safety_doc)]
pub mod driver_sys;
#[cfg(feature = "runtime")]
#[allow(clippy::missing_safety_doc)]
pub mod runtime_sys;

#[cfg(feature = "cublas")]
#[allow(clippy::missing_safety_doc)]
pub mod cublas_sys;
#[cfg(feature = "cublaslt")]
#[allow(clippy::missing_safety_doc)]
pub mod cublaslt_sys;
#[cfg(feature = "cublasxt")]
#[allow(clippy::missing_safety_doc)]
pub mod cublasxt_sys;

#[cfg(feature = "nvptx-compiler")]
#[allow(clippy::missing_safety_doc)]
pub mod nvptx_compiler_sys;
#[cfg(feature = "nvvm")]
#[allow(clippy::missing_safety_doc)]
pub mod nvvm_sys;
