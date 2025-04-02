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
    #[cfg(not(cudnn9))]
    AllocFailed,
    /// An incorrect value or parameter was passed to the function.
    BadParam,
    /// An internal cuDNN operation failed.
    InternalError,
    InvalidValue,
    /// The function requires a feature absent from the current GPU device. Note that cuDNN only
    /// supports devices with compute capabilities greater than or equal to 3.0.
    #[cfg(not(cudnn9))]
    ArchMismatch,
    /// An access to GPU memory space failed, which is usually caused by a failure to bind a
    /// texture.
    #[cfg(not(cudnn9))]
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
    #[cfg(not(cudnn9))]
    RuntimePrerequisiteMissing,
    /// Some tasks in the user stream are not completed.
    RuntimeInProgress,
    /// Numerical overflow occurred during the GPU kernel execution.
    RuntimeFpOverflow,
    #[cfg(not(cudnn9))]
    VersionMismatch,
}

impl CudnnError {
    /// Converts the `CudnnError` into the corresponding raw variant.
    pub fn into_raw(self) -> cudnn_sys::cudnnStatus_t {
        use cudnn_sys::cudnnStatus_t::*;
        match self {
            CudnnError::NotInitialized => CUDNN_STATUS_NOT_INITIALIZED,
            #[cfg(not(cudnn9))]
            CudnnError::AllocFailed => CUDNN_STATUS_ALLOC_FAILED,
            CudnnError::BadParam => CUDNN_STATUS_BAD_PARAM,
            CudnnError::InternalError => CUDNN_STATUS_INTERNAL_ERROR,
            CudnnError::InvalidValue => CUDNN_STATUS_INVALID_VALUE,
            #[cfg(not(cudnn9))]
            CudnnError::ArchMismatch => CUDNN_STATUS_ARCH_MISMATCH,
            #[cfg(not(cudnn9))]
            CudnnError::MappingError => CUDNN_STATUS_MAPPING_ERROR,
            CudnnError::ExecutionFailed => CUDNN_STATUS_EXECUTION_FAILED,
            CudnnError::NotSupported => CUDNN_STATUS_NOT_SUPPORTED,
            CudnnError::LicenseError => CUDNN_STATUS_LICENSE_ERROR,
            #[cfg(not(cudnn9))]
            CudnnError::RuntimePrerequisiteMissing => CUDNN_STATUS_RUNTIME_PREREQUISITE_MISSING,
            CudnnError::RuntimeInProgress => CUDNN_STATUS_RUNTIME_IN_PROGRESS,
            CudnnError::RuntimeFpOverflow => CUDNN_STATUS_RUNTIME_FP_OVERFLOW,
            #[cfg(not(cudnn9))]
            CudnnError::VersionMismatch => CUDNN_STATUS_VERSION_MISMATCH,
        }
    }
}

impl Display for CudnnError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        unsafe {
            let ptr = cudnn_sys::cudnnGetErrorString(self.into_raw());
            let cow = CStr::from_ptr(ptr).to_string_lossy();
            f.write_str(cow.as_ref())
        }
    }
}

impl Error for CudnnError {}

pub trait IntoResult {
    fn into_result(self) -> Result<(), CudnnError>;
}

impl IntoResult for cudnn_sys::cudnnStatus_t {
    /// Converts the raw status into a result.
    fn into_result(self) -> Result<(), CudnnError> {
        use cudnn_sys::cudnnStatus_t::*;

        Err(match self {
            CUDNN_STATUS_SUCCESS => return Ok(()),
            CUDNN_STATUS_NOT_INITIALIZED => CudnnError::NotInitialized,
            #[cfg(not(cudnn9))]
            CUDNN_STATUS_ALLOC_FAILED => CudnnError::AllocFailed,
            CUDNN_STATUS_BAD_PARAM => CudnnError::BadParam,
            CUDNN_STATUS_INTERNAL_ERROR => CudnnError::InternalError,
            CUDNN_STATUS_INVALID_VALUE => CudnnError::InvalidValue,
            #[cfg(not(cudnn9))]
            CUDNN_STATUS_ARCH_MISMATCH => CudnnError::ArchMismatch,
            #[cfg(not(cudnn9))]
            CUDNN_STATUS_MAPPING_ERROR => CudnnError::MappingError,
            CUDNN_STATUS_EXECUTION_FAILED => CudnnError::ExecutionFailed,
            CUDNN_STATUS_NOT_SUPPORTED => CudnnError::NotSupported,
            CUDNN_STATUS_LICENSE_ERROR => CudnnError::LicenseError,
            #[cfg(not(cudnn9))]
            CUDNN_STATUS_RUNTIME_PREREQUISITE_MISSING => CudnnError::RuntimePrerequisiteMissing,
            CUDNN_STATUS_RUNTIME_IN_PROGRESS => CudnnError::RuntimeInProgress,
            CUDNN_STATUS_RUNTIME_FP_OVERFLOW => CudnnError::RuntimeFpOverflow,
            #[cfg(not(cudnn9))]
            CUDNN_STATUS_VERSION_MISMATCH => CudnnError::VersionMismatch,
            // TODO(adamcavendish): implement cuDNN 9 error codes.
            _ => todo!(),
        })
    }
}
