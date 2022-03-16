use crate::sys;
use std::{error::Error, ffi::CStr, fmt::Display};

/// Enum encapsulating function status returns. All cuDNN library functions return their status.
///
/// cuDNN [docs](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnStatus_t)
/// may offer additional information about the APi behavior.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CudnnError {
    /// The cuDNN library was not initialized properly.
    ///
    /// This error is usually returned when a call to [`crate::CudnnContext::new()`] fails or when
    /// `CudnnContext::new()` has not been called prior to calling another cuDNN routine. In the
    /// former case, it is usually due to an error in the CUDA Runtime API called by such a function
    /// or by an error in the hardware setup.
    NotInitialized,
    /// Resource allocation failed inside the cuDNN library. This is usually caused by an internal
    /// `cudaMalloc()` failure.
    AllocFailed,
    /// An incorrect value or parameter was passed to the function.
    BadParam,
    /// An internal cuDNN operation failed.
    InternalError,
    InvalidValue,
    /// The function requires a feature absent from the current GPU device. Note that cuDNN only
    /// supports devices with compute capabilities greater than or equal to 3.0.
    ArchMismatch,
    /// An access to GPU memory space failed, which is usually caused by a failure to bind a
    /// texture.
    MappingError,
    /// The GPU program failed to execute. This is usually caused by a failure to launch some
    /// cuDNN kernel on the GPU, which can occur for multiple reasons.
    ExecutionFailed,
    /// The functionality requested is not presently supported by cuDNN.
    NotSupported,
    /// The functionality requested requires some license and an error was detected when trying to
    /// check the current licensing. This error can happen if the license is not present or is
    /// expired or if the environment variable `NVIDIA_LICENSE_FILE` is not set properly.
    LicenseError,
    /// A runtime library required by cuDNN cannot be found in the predefined search paths.
    /// These libraries are libcuda.so (nvcuda.dll) and libnvrtc.so
    /// (nvrtc64_Major Release Version Minor Release Version_0.dll and
    /// nvrtc-builtins64_Major Release Version Minor Release Version.dll).
    RuntimePrerequisiteMissing,
    /// Some tasks in the user stream are not completed.
    RuntimeInProgress,
    /// Numerical overflow occurred during the GPU kernel execution.
    RuntimeFpOverflow,
    VersionMismatch,
}

impl CudnnError {
    /// Converts the `CudnnError` into the corresponding raw variant.
    pub fn into_raw(self) -> sys::cudnnStatus_t {
        match self {
            CudnnError::NotInitialized => sys::cudnnStatus_t::CUDNN_STATUS_NOT_INITIALIZED,
            CudnnError::AllocFailed => sys::cudnnStatus_t::CUDNN_STATUS_ALLOC_FAILED,
            CudnnError::BadParam => sys::cudnnStatus_t::CUDNN_STATUS_BAD_PARAM,
            CudnnError::InternalError => sys::cudnnStatus_t::CUDNN_STATUS_INTERNAL_ERROR,
            CudnnError::InvalidValue => sys::cudnnStatus_t::CUDNN_STATUS_INVALID_VALUE,
            CudnnError::ArchMismatch => sys::cudnnStatus_t::CUDNN_STATUS_ARCH_MISMATCH,
            CudnnError::MappingError => sys::cudnnStatus_t::CUDNN_STATUS_MAPPING_ERROR,
            CudnnError::ExecutionFailed => sys::cudnnStatus_t::CUDNN_STATUS_EXECUTION_FAILED,
            CudnnError::NotSupported => sys::cudnnStatus_t::CUDNN_STATUS_NOT_SUPPORTED,
            CudnnError::LicenseError => sys::cudnnStatus_t::CUDNN_STATUS_LICENSE_ERROR,
            CudnnError::RuntimePrerequisiteMissing => {
                sys::cudnnStatus_t::CUDNN_STATUS_RUNTIME_PREREQUISITE_MISSING
            }
            CudnnError::RuntimeInProgress => sys::cudnnStatus_t::CUDNN_STATUS_RUNTIME_IN_PROGRESS,
            CudnnError::RuntimeFpOverflow => sys::cudnnStatus_t::CUDNN_STATUS_RUNTIME_FP_OVERFLOW,
            CudnnError::VersionMismatch => sys::cudnnStatus_t::CUDNN_STATUS_VERSION_MISMATCH,
        }
    }
}

impl Display for CudnnError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        unsafe {
            let ptr = sys::cudnnGetErrorString(self.into_raw());
            let cow = CStr::from_ptr(ptr).to_string_lossy();
            f.write_str(cow.as_ref())
        }
    }
}

impl Error for CudnnError {}

pub trait IntoResult {
    fn into_result(self) -> Result<(), CudnnError>;
}

impl IntoResult for sys::cudnnStatus_t {
    /// Converts the raw status into a result.
    fn into_result(self) -> Result<(), CudnnError> {
        Err(match self {
            sys::cudnnStatus_t::CUDNN_STATUS_SUCCESS => return Ok(()),
            sys::cudnnStatus_t::CUDNN_STATUS_NOT_INITIALIZED => CudnnError::NotInitialized,
            sys::cudnnStatus_t::CUDNN_STATUS_ALLOC_FAILED => CudnnError::AllocFailed,
            sys::cudnnStatus_t::CUDNN_STATUS_BAD_PARAM => CudnnError::BadParam,
            sys::cudnnStatus_t::CUDNN_STATUS_INTERNAL_ERROR => CudnnError::InternalError,
            sys::cudnnStatus_t::CUDNN_STATUS_INVALID_VALUE => CudnnError::InvalidValue,
            sys::cudnnStatus_t::CUDNN_STATUS_ARCH_MISMATCH => CudnnError::ArchMismatch,
            sys::cudnnStatus_t::CUDNN_STATUS_MAPPING_ERROR => CudnnError::MappingError,
            sys::cudnnStatus_t::CUDNN_STATUS_EXECUTION_FAILED => CudnnError::ExecutionFailed,
            sys::cudnnStatus_t::CUDNN_STATUS_NOT_SUPPORTED => CudnnError::NotSupported,
            sys::cudnnStatus_t::CUDNN_STATUS_LICENSE_ERROR => CudnnError::LicenseError,
            sys::cudnnStatus_t::CUDNN_STATUS_RUNTIME_PREREQUISITE_MISSING => {
                CudnnError::RuntimePrerequisiteMissing
            }
            sys::cudnnStatus_t::CUDNN_STATUS_RUNTIME_IN_PROGRESS => CudnnError::RuntimeInProgress,
            sys::cudnnStatus_t::CUDNN_STATUS_RUNTIME_FP_OVERFLOW => CudnnError::RuntimeFpOverflow,
            sys::cudnnStatus_t::CUDNN_STATUS_VERSION_MISMATCH => CudnnError::VersionMismatch,
        })
    }
}
