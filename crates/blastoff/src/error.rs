use crate::sys;
use cust::error::CudaError;
use std::{ffi::CStr, fmt::Display};

/// Result that contains the un-dropped value on error.
pub type DropResult<T> = std::result::Result<(), (CublasError, T)>;

#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CublasError {
    NotInitialized,
    AllocFailed,
    InvalidValue,
    ArchMismatch,
    MappingError,
    ExecutionFailed,
    InternalError,
    NotSupported,
    LicenseError,
}

impl std::error::Error for CublasError {}

impl Display for CublasError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        unsafe {
            let ptr = sys::v2::cublasGetStatusString(self.into_raw());
            let cow = CStr::from_ptr(ptr).to_string_lossy();
            f.write_str(cow.as_ref())
        }
    }
}

pub trait ToResult {
    fn to_result(self) -> Result<(), CublasError>;
}

impl ToResult for sys::v2::cublasStatus_t {
    fn to_result(self) -> Result<(), CublasError> {
        use CublasError::*;

        Err(match self {
            sys::v2::cublasStatus_t::CUBLAS_STATUS_SUCCESS => return Ok(()),
            sys::v2::cublasStatus_t::CUBLAS_STATUS_NOT_INITIALIZED => NotInitialized,
            sys::v2::cublasStatus_t::CUBLAS_STATUS_ALLOC_FAILED => AllocFailed,
            sys::v2::cublasStatus_t::CUBLAS_STATUS_INVALID_VALUE => InvalidValue,
            sys::v2::cublasStatus_t::CUBLAS_STATUS_ARCH_MISMATCH => ArchMismatch,
            sys::v2::cublasStatus_t::CUBLAS_STATUS_MAPPING_ERROR => MappingError,
            sys::v2::cublasStatus_t::CUBLAS_STATUS_EXECUTION_FAILED => ExecutionFailed,
            sys::v2::cublasStatus_t::CUBLAS_STATUS_INTERNAL_ERROR => InternalError,
            sys::v2::cublasStatus_t::CUBLAS_STATUS_NOT_SUPPORTED => NotSupported,
            sys::v2::cublasStatus_t::CUBLAS_STATUS_LICENSE_ERROR => LicenseError,
        })
    }
}

impl CublasError {
    pub fn into_raw(self) -> sys::v2::cublasStatus_t {
        use CublasError::*;

        match self {
            NotInitialized => sys::v2::cublasStatus_t::CUBLAS_STATUS_NOT_INITIALIZED,
            AllocFailed => sys::v2::cublasStatus_t::CUBLAS_STATUS_ALLOC_FAILED,
            InvalidValue => sys::v2::cublasStatus_t::CUBLAS_STATUS_INVALID_VALUE,
            ArchMismatch => sys::v2::cublasStatus_t::CUBLAS_STATUS_ARCH_MISMATCH,
            MappingError => sys::v2::cublasStatus_t::CUBLAS_STATUS_MAPPING_ERROR,
            ExecutionFailed => sys::v2::cublasStatus_t::CUBLAS_STATUS_EXECUTION_FAILED,
            InternalError => sys::v2::cublasStatus_t::CUBLAS_STATUS_INTERNAL_ERROR,
            NotSupported => sys::v2::cublasStatus_t::CUBLAS_STATUS_NOT_SUPPORTED,
            LicenseError => sys::v2::cublasStatus_t::CUBLAS_STATUS_LICENSE_ERROR,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Error {
    Cublas(CublasError),
    Cuda(CudaError),
}

impl From<CublasError> for Error {
    fn from(err: CublasError) -> Self {
        Self::Cublas(err)
    }
}

impl From<CudaError> for Error {
    fn from(err: CudaError) -> Self {
        Self::Cuda(err)
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Cublas(e) => Some(e),
            Self::Cuda(e) => Some(e),
        }
    }
}

impl Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cublas(_) => write!(f, "cuBLAS error"),
            Self::Cuda(_) => write!(f, "CUDA error"),
        }
    }
}
