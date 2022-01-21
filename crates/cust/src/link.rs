//! Functions for linking together multiple PTX files into a module.

use std::mem::MaybeUninit;
use std::ptr::null_mut;

use crate::sys as cuda;

use crate::error::{CudaResult, ToResult};

static UNNAMED: &str = "\0";

/// A linker used to link together PTX files into a single module.
#[derive(Debug)]
pub struct Linker {
    raw: cuda::CUlinkState,
}

unsafe impl Send for Linker {}
unsafe impl Sync for Linker {}

impl Linker {
    /// Creates a new linker.
    pub fn new() -> CudaResult<Self> {
        // per the docs, cuda expects the options pointers to last as long as CULinkState.
        // Therefore we use box to alloc the memory for us, then into_raw it so we now have ownership
        // of the memory (and dont have any aliasing requirements attached either).

        unsafe {
            let mut raw = MaybeUninit::uninit();
            cuda::cuLinkCreate_v2(0, null_mut(), null_mut(), raw.as_mut_ptr()).to_result()?;
            Ok(Self {
                raw: raw.assume_init(),
            })
        }
    }

    // TODO(RDambrosio016): Support PTX compiler options and decide whether we should expose
    // them as a separate crate or as part of cust.

    /// Add some PTX assembly string to be linked in. The PTX code will be
    /// compiled into cubin by CUDA then linked in.
    ///
    /// # Returns
    ///
    /// Returns an error if the PTX is invalid, cuda is out of memory, or the PTX
    /// is of an unsupported version.
    pub fn add_ptx(&mut self, ptx: impl AsRef<str>) -> CudaResult<()> {
        let ptx = ptx.as_ref();

        unsafe {
            cuda::cuLinkAddData_v2(
                self.raw,
                cuda::CUjitInputType::CU_JIT_INPUT_PTX,
                // cuda_sys wants *mut but from the API docs we know we retain ownership so
                // this cast is sound.
                ptx.as_ptr() as *mut _,
                ptx.len(),
                UNNAMED.as_ptr().cast(),
                0,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            )
            .to_result()
        }
    }

    /// Add some cubin (CUDA binary) to be linked in.
    ///
    /// # Returns
    ///
    /// Returns an error if the cubin is invalid or CUDA is out of memory.
    pub fn add_cubin(&mut self, cubin: impl AsRef<[u8]>) -> CudaResult<()> {
        let cubin = cubin.as_ref();

        unsafe {
            cuda::cuLinkAddData_v2(
                self.raw,
                cuda::CUjitInputType::CU_JIT_INPUT_CUBIN,
                // cuda_sys wants *mut but from the API docs we know we retain ownership so
                // this cast is sound.
                cubin.as_ptr() as *mut _,
                cubin.len(),
                UNNAMED.as_ptr().cast(),
                0,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            )
            .to_result()
        }
    }

    /// Add a fatbin (Fat Binary) to be linked in.
    ///
    /// # Returns
    ///
    /// Returns an error if the fatbin is invalid or CUDA is out of memory.
    pub fn add_fatbin(&mut self, fatbin: impl AsRef<[u8]>) -> CudaResult<()> {
        let fatbin = fatbin.as_ref();

        unsafe {
            cuda::cuLinkAddData_v2(
                self.raw,
                cuda::CUjitInputType::CU_JIT_INPUT_FATBINARY,
                // cuda_sys wants *mut but from the API docs we know we retain ownership so
                // this cast is sound.
                fatbin.as_ptr() as *mut _,
                fatbin.len(),
                UNNAMED.as_ptr().cast(),
                0,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            )
            .to_result()
        }
    }

    /// Runs the linker to generate the final cubin bytes. Also returns a duration
    /// for how long it took to run the linker.
    pub fn complete(self) -> CudaResult<Vec<u8>> {
        let mut cubin = MaybeUninit::uninit();
        let mut size = MaybeUninit::uninit();

        unsafe {
            cuda::cuLinkComplete(self.raw, cubin.as_mut_ptr(), size.as_mut_ptr()).to_result()?;
            // docs say that CULinkState owns the data, so clone it out before we destroy ourselves.
            let cubin = cubin.assume_init() as *const u8;
            let size = size.assume_init();
            let slice = std::slice::from_raw_parts(cubin, size);
            let mut vec = Vec::with_capacity(size);
            vec.extend_from_slice(slice);

            Ok(vec)
        }
    }
}

impl Drop for Linker {
    fn drop(&mut self) {
        unsafe { cuda::cuLinkDestroy(self.raw) };
    }
}
