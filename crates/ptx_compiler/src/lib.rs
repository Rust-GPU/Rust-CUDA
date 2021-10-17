//! Raw and High level bindings to the CUDA NVPTX compiler used to compile PTX to
//! cubin files.

use std::mem::MaybeUninit;

#[allow(warnings)]
pub mod sys;

trait ToResult {
    fn to_result(self) -> Result<(), NvptxError>;
}

impl ToResult for sys::nvPTXCompileResult {
    fn to_result(self) -> Result<(), NvptxError> {
        match self {
            sys::nvPTXCompileResult_NVPTXCOMPILE_SUCCESS => Ok(()),
            sys::nvPTXCompileResult_NVPTXCOMPILE_ERROR_INVALID_INPUT => {
                Err(NvptxError::InvalidInput)
            }
            sys::nvPTXCompileResult_NVPTXCOMPILE_ERROR_COMPILATION_FAILURE => {
                Err(NvptxError::CompilationFailure)
            }
            sys::nvPTXCompileResult_NVPTXCOMPILE_ERROR_INTERNAL => Err(NvptxError::Internal),
            sys::nvPTXCompileResult_NVPTXCOMPILE_ERROR_OUT_OF_MEMORY => {
                Err(NvptxError::OutOfMemory)
            }
            sys::nvPTXCompileResult_NVPTXCOMPILE_ERROR_UNSUPPORTED_PTX_VERSION => {
                Err(NvptxError::UnsupportedPtxVersion)
            }
            // these two are statically prevented so they should never happen
            sys::nvPTXCompileResult_NVPTXCOMPILE_ERROR_COMPILER_INVOCATION_INCOMPLETE => {
                unreachable!("nvptx yielded an incomplete invocation error")
            }
            sys::nvPTXCompileResult_NVPTXCOMPILE_ERROR_INVALID_COMPILER_HANDLE => {
                unreachable!("nvptx yielded an invalid handle err")
            }
            _ => unreachable!(),
        }
    }
}

pub type NvptxResult<T> = Result<T, NvptxError>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NvptxError {
    InvalidInput,
    CompilationFailure,
    Internal,
    OutOfMemory,
    UnsupportedPtxVersion,
}

#[repr(transparent)]
#[derive(Debug)]
pub struct NvptxCompiler {
    raw: sys::nvPTXCompilerHandle,
}

impl NvptxCompiler {
    /// Create a new compiler from a ptx string.
    pub fn new(ptx: impl AsRef<str>) -> NvptxResult<Self> {
        let ptx = ptx.as_ref();
        let mut raw = MaybeUninit::uninit();

        unsafe {
            sys::nvPTXCompilerCreate(raw.as_mut_ptr(), ptx.len() as u64, ptx.as_ptr().cast())
                .to_result()?;
            let raw = raw.assume_init();
            Ok(Self { raw })
        }
    }
}

impl Drop for NvptxCompiler {
    fn drop(&mut self) {
        unsafe {
            sys::nvPTXCompilerDestroy(&mut self.raw as *mut _)
                .to_result()
                .expect("failed to destroy nvptx compiler");
        }
    }
}

#[derive(Debug)]
pub struct CompilerFailure {
    pub error: NvptxError,
    handle: sys::nvPTXCompilerHandle,
}

impl Drop for CompilerFailure {
    fn drop(&mut self) {
        unsafe {
            sys::nvPTXCompilerDestroy(&mut self.handle as *mut _)
                .to_result()
                .expect("failed to destroy nvptx compiler failure");
        }
    }
}

impl CompilerFailure {
    pub fn error_log(&self) -> NvptxResult<String> {
        let mut size = MaybeUninit::uninit();
        unsafe {
            sys::nvPTXCompilerGetErrorLogSize(self.handle, size.as_mut_ptr()).to_result()?;
            let size = size.assume_init() as usize;
            let mut vec = Vec::with_capacity(size);
            sys::nvPTXCompilerGetErrorLog(self.handle, vec.as_mut_ptr() as *mut i8).to_result()?;
            vec.set_len(size);
            Ok(String::from_utf8_lossy(&vec).to_string())
        }
    }
}

/// The result of a compiled program
#[derive(Debug)]
pub struct CompiledProgram {
    pub cubin: Vec<u8>,
    handle: sys::nvPTXCompilerHandle,
}

impl Drop for CompiledProgram {
    fn drop(&mut self) {
        unsafe {
            sys::nvPTXCompilerDestroy(&mut self.handle as *mut _)
                .to_result()
                .expect("failed to destroy nvptx compiled program");
        }
    }
}

impl CompiledProgram {
    pub fn info_log(&self) -> NvptxResult<String> {
        let mut size = MaybeUninit::uninit();
        unsafe {
            sys::nvPTXCompilerGetInfoLogSize(self.handle, size.as_mut_ptr()).to_result()?;
            let size = size.assume_init() as usize;
            let mut vec = Vec::with_capacity(size);
            sys::nvPTXCompilerGetInfoLog(self.handle, vec.as_mut_ptr() as *mut i8).to_result()?;
            vec.set_len(size);
            Ok(String::from_utf8_lossy(&vec).to_string())
        }
    }
}
