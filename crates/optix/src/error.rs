use std::{
    ffi::CStr,
    fmt::{Debug, Display},
};

use cust::error::CudaError;

/// Any error which may occur when executing an OptiX function.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OptixError {
    InvalidValue,
    HostOutOfMemory,
    InvalidOperation,
    FileIoError,
    InvalidFileFormat,
    DiskCacheInvalidPath,
    DiskCachePermissionError,
    DiskCacheDatabaseError,
    DiskCacheInvalidData,
    LaunchFailure,
    InvalidDeviceContext,
    CudaNotInitialized,
    ValidationFailure,
    InvalidPtx,
    InvalidLaunchParameter,
    InvalidPayloadAccess,
    InvalidAttributeAccess,
    InvalidFunctionUse,
    InvalidFunctionArguments,
    PipelineOutOfConstantMemory,
    PipelineLinkError,
    IllegalDuringTaskExecute,
    InternalCompilerError,
    DenoiserModelNotSet,
    DenoiserNotInitialized,
    AccelNotCompatible,
    NotSupported,
    UnsupportedAbiVersion,
    FunctionTableSizeMismatch,
    InvalidEntryFunctionOptions,
    LibraryNotFound,
    EntrySymbolNotFound,
    LibraryUnloadFailure,
    CudaError,
    InternalError,
    Unknown,

    // optix stubs doesnt do this it just segfaults :)
    OptixNotInitialized,
}

impl OptixError {
    pub fn to_raw(self) -> optix_sys::OptixResult {
        use OptixError::*;

        match self {
            InvalidValue => optix_sys::OptixResult::OPTIX_ERROR_INVALID_VALUE,
            HostOutOfMemory => optix_sys::OptixResult::OPTIX_ERROR_HOST_OUT_OF_MEMORY,
            InvalidOperation => optix_sys::OptixResult::OPTIX_ERROR_INVALID_OPERATION,
            FileIoError => optix_sys::OptixResult::OPTIX_ERROR_FILE_IO_ERROR,
            InvalidFileFormat => optix_sys::OptixResult::OPTIX_ERROR_INVALID_FILE_FORMAT,
            DiskCacheInvalidPath => optix_sys::OptixResult::OPTIX_ERROR_DISK_CACHE_INVALID_PATH,
            DiskCachePermissionError => {
                optix_sys::OptixResult::OPTIX_ERROR_DISK_CACHE_PERMISSION_ERROR
            }
            DiskCacheDatabaseError => optix_sys::OptixResult::OPTIX_ERROR_DISK_CACHE_DATABASE_ERROR,
            DiskCacheInvalidData => optix_sys::OptixResult::OPTIX_ERROR_DISK_CACHE_INVALID_DATA,
            LaunchFailure => optix_sys::OptixResult::OPTIX_ERROR_LAUNCH_FAILURE,
            InvalidDeviceContext => optix_sys::OptixResult::OPTIX_ERROR_INVALID_DEVICE_CONTEXT,
            CudaNotInitialized => optix_sys::OptixResult::OPTIX_ERROR_CUDA_NOT_INITIALIZED,
            ValidationFailure => optix_sys::OptixResult::OPTIX_ERROR_VALIDATION_FAILURE,
            InvalidPtx => optix_sys::OptixResult::OPTIX_ERROR_INVALID_INPUT,
            InvalidLaunchParameter => optix_sys::OptixResult::OPTIX_ERROR_INVALID_LAUNCH_PARAMETER,
            InvalidPayloadAccess => optix_sys::OptixResult::OPTIX_ERROR_INVALID_PAYLOAD_ACCESS,
            InvalidAttributeAccess => optix_sys::OptixResult::OPTIX_ERROR_INVALID_ATTRIBUTE_ACCESS,
            InvalidFunctionUse => optix_sys::OptixResult::OPTIX_ERROR_INVALID_FUNCTION_USE,
            InvalidFunctionArguments => {
                optix_sys::OptixResult::OPTIX_ERROR_INVALID_FUNCTION_ARGUMENTS
            }
            PipelineOutOfConstantMemory => {
                optix_sys::OptixResult::OPTIX_ERROR_PIPELINE_OUT_OF_CONSTANT_MEMORY
            }
            PipelineLinkError => optix_sys::OptixResult::OPTIX_ERROR_PIPELINE_LINK_ERROR,
            IllegalDuringTaskExecute => {
                optix_sys::OptixResult::OPTIX_ERROR_ILLEGAL_DURING_TASK_EXECUTE
            }
            InternalCompilerError => optix_sys::OptixResult::OPTIX_ERROR_INTERNAL_COMPILER_ERROR,
            DenoiserModelNotSet => optix_sys::OptixResult::OPTIX_ERROR_DENOISER_MODEL_NOT_SET,
            DenoiserNotInitialized => optix_sys::OptixResult::OPTIX_ERROR_DENOISER_NOT_INITIALIZED,
            AccelNotCompatible => optix_sys::OptixResult::OPTIX_ERROR_NOT_COMPATIBLE,
            NotSupported => optix_sys::OptixResult::OPTIX_ERROR_NOT_SUPPORTED,
            UnsupportedAbiVersion => optix_sys::OptixResult::OPTIX_ERROR_UNSUPPORTED_ABI_VERSION,
            FunctionTableSizeMismatch => {
                optix_sys::OptixResult::OPTIX_ERROR_FUNCTION_TABLE_SIZE_MISMATCH
            }
            InvalidEntryFunctionOptions => {
                optix_sys::OptixResult::OPTIX_ERROR_INVALID_ENTRY_FUNCTION_OPTIONS
            }
            LibraryNotFound => optix_sys::OptixResult::OPTIX_ERROR_LIBRARY_NOT_FOUND,
            EntrySymbolNotFound => optix_sys::OptixResult::OPTIX_ERROR_ENTRY_SYMBOL_NOT_FOUND,
            LibraryUnloadFailure => optix_sys::OptixResult::OPTIX_ERROR_LIBRARY_UNLOAD_FAILURE,
            CudaError => optix_sys::OptixResult::OPTIX_ERROR_CUDA_ERROR,
            InternalError => optix_sys::OptixResult::OPTIX_ERROR_INTERNAL_ERROR,
            Unknown => optix_sys::OptixResult::OPTIX_ERROR_UNKNOWN,
            // close enough
            OptixNotInitialized => optix_sys::OptixResult::OPTIX_ERROR_CUDA_NOT_INITIALIZED,
        }
    }
}

impl From<CudaError> for OptixError {
    fn from(_: CudaError) -> Self {
        Self::CudaError
    }
}

impl From<OptixError> for CudaError {
    fn from(_: OptixError) -> Self {
        CudaError::OptixError
    }
}

impl Display for OptixError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        unsafe {
            if *self == OptixError::OptixNotInitialized {
                return f.write_str("OptiX was not initialized");
            }
            // optix_stubs special cases this function if optix is not initialized so we dont need to
            // optix_call this.
            let ptr = optix_sys::optixGetErrorString(self.to_raw());
            let cow = CStr::from_ptr(ptr).to_string_lossy();
            f.write_str(cow.as_ref())
        }
    }
}

impl std::error::Error for OptixError {}

// pub type OptixResult<T> = Result<T, OptixError>;

pub trait ToResult {
    fn to_result(self) -> Result<(), OptixError>;
}

impl ToResult for optix_sys::OptixResult {
    fn to_result(self) -> Result<(), OptixError> {
        use OptixError::*;

        Err(match self {
            optix_sys::OptixResult::OPTIX_SUCCESS => return Ok(()),
            optix_sys::OptixResult::OPTIX_ERROR_INVALID_VALUE => InvalidValue,
            optix_sys::OptixResult::OPTIX_ERROR_HOST_OUT_OF_MEMORY => HostOutOfMemory,
            optix_sys::OptixResult::OPTIX_ERROR_INVALID_OPERATION => InvalidOperation,
            optix_sys::OptixResult::OPTIX_ERROR_FILE_IO_ERROR => FileIoError,
            optix_sys::OptixResult::OPTIX_ERROR_INVALID_FILE_FORMAT => InvalidFileFormat,
            optix_sys::OptixResult::OPTIX_ERROR_DISK_CACHE_INVALID_PATH => DiskCacheInvalidPath,
            optix_sys::OptixResult::OPTIX_ERROR_DISK_CACHE_PERMISSION_ERROR => {
                DiskCachePermissionError
            }
            optix_sys::OptixResult::OPTIX_ERROR_DISK_CACHE_DATABASE_ERROR => DiskCacheDatabaseError,
            optix_sys::OptixResult::OPTIX_ERROR_DISK_CACHE_INVALID_DATA => DiskCacheInvalidData,
            optix_sys::OptixResult::OPTIX_ERROR_LAUNCH_FAILURE => LaunchFailure,
            optix_sys::OptixResult::OPTIX_ERROR_INVALID_DEVICE_CONTEXT => InvalidDeviceContext,
            optix_sys::OptixResult::OPTIX_ERROR_CUDA_NOT_INITIALIZED => CudaNotInitialized,
            optix_sys::OptixResult::OPTIX_ERROR_VALIDATION_FAILURE => ValidationFailure,
            optix_sys::OptixResult::OPTIX_ERROR_INVALID_INPUT => InvalidPtx,
            optix_sys::OptixResult::OPTIX_ERROR_INVALID_LAUNCH_PARAMETER => InvalidLaunchParameter,
            optix_sys::OptixResult::OPTIX_ERROR_INVALID_PAYLOAD_ACCESS => InvalidPayloadAccess,
            optix_sys::OptixResult::OPTIX_ERROR_INVALID_ATTRIBUTE_ACCESS => InvalidAttributeAccess,
            optix_sys::OptixResult::OPTIX_ERROR_INVALID_FUNCTION_USE => InvalidFunctionUse,
            optix_sys::OptixResult::OPTIX_ERROR_INVALID_FUNCTION_ARGUMENTS => {
                InvalidFunctionArguments
            }
            optix_sys::OptixResult::OPTIX_ERROR_PIPELINE_OUT_OF_CONSTANT_MEMORY => {
                PipelineOutOfConstantMemory
            }
            optix_sys::OptixResult::OPTIX_ERROR_PIPELINE_LINK_ERROR => PipelineLinkError,
            optix_sys::OptixResult::OPTIX_ERROR_ILLEGAL_DURING_TASK_EXECUTE => {
                IllegalDuringTaskExecute
            }
            optix_sys::OptixResult::OPTIX_ERROR_INTERNAL_COMPILER_ERROR => InternalCompilerError,
            optix_sys::OptixResult::OPTIX_ERROR_DENOISER_MODEL_NOT_SET => DenoiserModelNotSet,
            optix_sys::OptixResult::OPTIX_ERROR_DENOISER_NOT_INITIALIZED => DenoiserNotInitialized,
            optix_sys::OptixResult::OPTIX_ERROR_NOT_COMPATIBLE => AccelNotCompatible,
            optix_sys::OptixResult::OPTIX_ERROR_NOT_SUPPORTED => NotSupported,
            optix_sys::OptixResult::OPTIX_ERROR_UNSUPPORTED_ABI_VERSION => UnsupportedAbiVersion,
            optix_sys::OptixResult::OPTIX_ERROR_FUNCTION_TABLE_SIZE_MISMATCH => {
                FunctionTableSizeMismatch
            }
            optix_sys::OptixResult::OPTIX_ERROR_INVALID_ENTRY_FUNCTION_OPTIONS => {
                InvalidEntryFunctionOptions
            }
            optix_sys::OptixResult::OPTIX_ERROR_LIBRARY_NOT_FOUND => LibraryNotFound,
            optix_sys::OptixResult::OPTIX_ERROR_ENTRY_SYMBOL_NOT_FOUND => EntrySymbolNotFound,
            optix_sys::OptixResult::OPTIX_ERROR_LIBRARY_UNLOAD_FAILURE => LibraryUnloadFailure,
            optix_sys::OptixResult::OPTIX_ERROR_CUDA_ERROR => CudaError,
            optix_sys::OptixResult::OPTIX_ERROR_INTERNAL_ERROR => InternalError,
            optix_sys::OptixResult::OPTIX_ERROR_UNKNOWN => Unknown,
            value => panic!("Unhandled OptixResult value {:?}", value),
        })
    }
}

#[derive(Debug)]
pub enum Error {
    Optix(OptixError),
    Cuda(CudaError),
    ModuleCreation { source: OptixError, log: String },
    ProgramGroupCreation { source: OptixError, log: String },
    PipelineCreation { source: OptixError, log: String },
    AccelUpdateMismatch,
    NulBytesInString,
    TooFewMotionKeys(usize),
}

impl From<OptixError> for Error {
    fn from(o: OptixError) -> Self {
        Self::Optix(o)
    }
}

impl From<CudaError> for Error {
    fn from(e: CudaError) -> Self {
        Self::Cuda(e)
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Optix(e) => Some(e),
            Self::Cuda(e) => Some(e),
            Self::ModuleCreation { source, .. } => Some(source),
            Self::ProgramGroupCreation { source, .. } => Some(source),
            Self::PipelineCreation { source, .. } => Some(source),
            Self::AccelUpdateMismatch => None,
            Self::NulBytesInString => None,
            Self::TooFewMotionKeys(_) => None,
        }
    }
}

impl Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Optix(_) => write!(f, "OptiX error"),
            Self::Cuda(_) => write!(f, "CUDA error"),
            Self::ModuleCreation { log, .. } => write!(f, "Module creation error: {}", log),
            Self::ProgramGroupCreation { log, .. } => {
                write!(f, "Program group creation error: {}", log)
            }
            Self::PipelineCreation { log, .. } => write!(f, "Pipeline creation error: {}", log),
            Self::AccelUpdateMismatch => write!(f, "Build inputs passed to DynamicAccel::update do not match the structure of those used to build the accel"),
            Self::NulBytesInString => write!(f, "The provided string contained nul bytes"),
            Self::TooFewMotionKeys(num) => write!(f, "Provided too few motion keys ({}) for transform. Must provide at least 2", num),
        }
    }
}
