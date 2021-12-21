use crate::{
    context::ContextFlags,
    device::Device,
    error::{CudaResult, ToResult},
    sys as cuda,
};
use std::mem::MaybeUninit;

#[derive(Debug)]
pub struct Context {
    inner: cuda::CUcontext,
    device: cuda::CUdevice,
}

impl Context {
    /// Retains the primary context for this device, incrementing the internal reference cycle
    /// that CUDA keeps track of. There is only one primary context associated with a device, multiple
    /// calls to this function with the same device will return the same internal context.
    ///
    /// This will **NOT** push the context to the stack, primary contexts do not interoperate
    /// with the context stack.
    pub fn new(device: &Device) -> CudaResult<Self> {
        let mut inner = MaybeUninit::uninit();
        unsafe {
            cuda::cuDevicePrimaryCtxRetain(inner.as_mut_ptr(), device.as_raw()).to_result()?;
            Ok(Self {
                inner: inner.assume_init(),
                device: device.as_raw(),
            })
        }
    }

    /// Resets the primary context associated with the device, freeing all allocations created
    /// inside of the context. You must make sure that nothing else is using the context or using
    /// CUDA on the device in general. For this reason, it is usually highly advised to not use
    /// this function.
    ///
    /// # Safety
    ///
    /// Nothing else should be using the primary context for this device, otherwise,
    /// spurious errors or segfaults will occur.
    pub unsafe fn reset(device: &Device) -> CudaResult<()> {
        cuda::cuDevicePrimaryCtxReset_v2(device.as_raw()).to_result()
    }

    /// Sets the flags for the device context, these flags will apply to any user of the primary
    /// context associated with this device.
    pub fn set_flags(&self, flags: ContextFlags) -> CudaResult<()> {
        unsafe { cuda::cuDevicePrimaryCtxSetFlags_v2(self.device, flags.bits()).to_result() }
    }

    /// Returns the raw handle to this context.
    pub fn as_raw(&self) -> cuda::CUcontext {
        self.inner
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        unsafe {
            cuda::cuDevicePrimaryCtxRelease_v2(self.device);
        }
    }
}
