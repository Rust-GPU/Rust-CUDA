use crate::{error::CudnnError, sys, IntoResult};
use cust::memory::GpuBuffer;

/// The descriptor of a dropout operation.
pub struct DropoutDescriptor<T>
where
    T: GpuBuffer<u8>,
{
    pub(crate) raw: sys::cudnnDropoutDescriptor_t,
    states: T,
}

impl<T> DropoutDescriptor<T>
where
    T: GpuBuffer<u8>,
{
    pub(crate) fn new(raw: sys::cudnnDropoutDescriptor_t, states: T) -> Self {
        Self { raw, states }
    }
}

impl<T> Drop for DropoutDescriptor<T>
where
    T: GpuBuffer<u8>,
{
    fn drop(&mut self) {
        unsafe {
            sys::cudnnDestroyDropoutDescriptor(self.raw);
        }
    }
}
