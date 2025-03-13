use crate::sys;
use cust::memory::GpuBuffer;

/// The descriptor of a dropout operation.
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct DropoutDescriptor<T>
where
    T: GpuBuffer<u8>,
{
    pub(crate) raw: sys::cudnnDropoutDescriptor_t,
    pub(crate) states: T,
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
