use crate::{sys, CudnnError, IntoResult};
use std::{mem::MaybeUninit, rc::Rc};

#[derive(PartialEq, Eq, Hash, Debug)]
pub(crate) struct Inner {
    pub(crate) raw: sys::cudnnBackendDescriptor_t,
}

impl Drop for Inner {
    fn drop(&mut self) {
        unsafe {
            sys::cudnnBackendDestroyDescriptor(self.raw);
        }
    }
}

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Descriptor(Rc<Inner>);

impl Descriptor {
    pub(crate) unsafe fn new(dtype: sys::cudnnBackendDescriptorType_t) -> Result<Self, CudnnError> {
        let mut raw = MaybeUninit::uninit();

        sys::cudnnBackendCreateDescriptor(dtype, raw.as_mut_ptr()).into_result()?;

        let raw = raw.assume_init();

        Ok(Self(Rc::new(Inner { raw })))
    }

    pub(crate) unsafe fn finalize(&mut self) -> Result<(), CudnnError> {
        sys::cudnnBackendFinalize(self.0.raw).into_result()
    }

    pub(crate) unsafe fn set_attribute<T: ?Sized>(
        &mut self,
        aname: sys::cudnnBackendAttributeName_t,
        atype: sys::cudnnBackendAttributeType_t,
        count: i64,
        val: &T,
    ) -> Result<(), CudnnError> {
        let ptr = val as *const T as *const std::ffi::c_void;

        sys::cudnnBackendSetAttribute(self.0.raw, aname, atype, count, ptr).into_result()
    }

    pub(crate) unsafe fn get_attribute_count(
        &self,
        aname: sys::cudnnBackendAttributeName_t,
        atype: sys::cudnnBackendAttributeType_t,
    ) -> Result<i64, CudnnError> {
        let mut count = MaybeUninit::<i64>::uninit();

        sys::cudnnBackendGetAttribute(
            self.0.raw,
            aname,
            atype,
            0,
            count.as_mut_ptr(),
            std::ptr::null_mut(),
        )
        .into_result()?;

        Ok(count.assume_init())
    }

    pub(crate) fn inner(&self) -> sys::cudnnBackendDescriptor_t {
        self.0.raw
    }
}
