//! Functions for linking together multiple PTX files into a module.

use std::{mem::MaybeUninit, time::Duration};

use crate::sys as cuda;

use crate::error::{CudaResult, ToResult};

static UNNAMED: &str = "\0";

/// A linker used to link together PTX files into a single module.
#[derive(Debug)]
pub struct Linker {
    raw: cuda::CUlinkState,
    duration_ptr: *mut *mut f32,
}

impl Linker {
    /// Creates a new linker.
    pub fn new() -> CudaResult<Self> {
        // per the docs, cuda expects the options pointers to last as long as CULinkState.
        // Therefore we use box to alloc the memory for us, then into_raw it so we now have ownership
        // of the memory (and dont have any aliasing requirements attached either).

        // technically it should be fine to alloc as Box<*mut f32> then dealloc with Box<Box<f32>> but
        // in the future rust may make this guarantee untrue so just alloc as Box<Box<f32>> then cast.
        let ptr = Box::into_raw(Box::new(Box::new(0.0f32))) as *mut *mut f32;

        // cuda shouldnt be modifying this but trust the bindings in that it wants a *mut ptr.
        let options = &mut [cuda::CUjit_option_enum::CU_JIT_WALL_TIME];
        unsafe {
            let mut raw = MaybeUninit::uninit();
            cuda::cuLinkCreate_v2(1, options.as_mut_ptr(), ptr.cast(), raw.as_mut_ptr())
                .to_result()?;
            Ok(Self {
                raw: raw.assume_init(),
                duration_ptr: ptr,
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
    pub fn complete(self) -> CudaResult<(Vec<u8>, Duration)> {
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

            // now that duration has been written to, retrieve it and deallocate it.
            let duration = **self.duration_ptr;
            // recreate a box from our pointer, which will take ownership
            // of it and then drop it immediately. This is sound because
            // complete consumes self.
            Box::from_raw(self.duration_ptr as *mut Box<f32>);

            // convert to nanos so we dont lose the decimal millisecs.
            let duration = Duration::from_nanos((duration * 1e6) as u64);
            Ok((vec, duration))
        }
    }
}

impl Drop for Linker {
    fn drop(&mut self) {
        unsafe { cuda::cuLinkDestroy(self.raw) };
    }
}
