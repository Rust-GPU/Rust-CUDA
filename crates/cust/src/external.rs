//! External memory and synchronization resources

use crate::error::{CudaResult, ToResult};
use crate::memory::{DeviceCopy, DevicePointer};

use cust_raw as sys;

#[repr(transparent)]
pub struct ExternalMemory(sys::CUexternalMemory);

impl ExternalMemory {
    // Import an external memory referenced by `fd` with `size`
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn import(fd: i32, size: usize) -> CudaResult<ExternalMemory> {
        let desc = sys::CUDA_EXTERNAL_MEMORY_HANDLE_DESC {
            type_: sys::CUexternalMemoryHandleType_enum::CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD,
            handle: sys::CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st__bindgen_ty_1 { fd },
            size: size as u64,
            flags: 0,
            reserved: Default::default(),
        };

        let mut memory: sys::CUexternalMemory = std::ptr::null_mut();

        sys::cuImportExternalMemory(&mut memory, &desc)
            .to_result()
            .map(|_| ExternalMemory(memory))
    }

    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn reimport(&mut self, fd: i32, size: usize) -> CudaResult<()> {
        // import new memory - this will call drop to destroy the old one
        *self = ExternalMemory::import(fd, size)?;

        Ok(())
    }

    // Map a buffer from this memory with `size` and `offset`
    pub fn mapped_buffer<T: DeviceCopy>(
        &self,
        size_in_bytes: usize,
        offset_in_bytes: usize,
    ) -> CudaResult<DevicePointer<T>> {
        let buffer_desc = sys::CUDA_EXTERNAL_MEMORY_BUFFER_DESC {
            flags: 0,
            size: size_in_bytes as u64,
            offset: offset_in_bytes as u64,
            reserved: Default::default(),
        };

        let mut dptr = 0;
        unsafe {
            sys::cuExternalMemoryGetMappedBuffer(&mut dptr, self.0, &buffer_desc)
                .to_result()
                .map(|_| DevicePointer::from_raw(dptr))
        }
    }
}

impl Drop for ExternalMemory {
    fn drop(&mut self) {
        unsafe {
            sys::cuDestroyExternalMemory(self.0).to_result().unwrap();
        }
    }
}
