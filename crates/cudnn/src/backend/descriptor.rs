use std::{mem::MaybeUninit, rc::Rc};

use crate::{CudnnError, IntoResult};

#[derive(PartialEq, Eq, Hash, Debug)]
pub(crate) struct Inner {
    pub(crate) raw: cudnn_sys::cudnnBackendDescriptor_t,
}

impl Drop for Inner {
    fn drop(&mut self) {
        unsafe {
            cudnn_sys::cudnnBackendDestroyDescriptor(self.raw);
        }
    }
}

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Descriptor(Rc<Inner>);

impl Descriptor {
    pub(crate) unsafe fn new(
        dtype: cudnn_sys::cudnnBackendDescriptorType_t,
    ) -> Result<Self, CudnnError> {
        let mut raw = MaybeUninit::uninit();

        cudnn_sys::cudnnBackendCreateDescriptor(dtype, raw.as_mut_ptr()).into_result()?;

        let raw = raw.assume_init();

        Ok(Self(Rc::new(Inner { raw })))
    }

    pub(crate) unsafe fn finalize(&mut self) -> Result<(), CudnnError> {
        cudnn_sys::cudnnBackendFinalize(self.0.raw).into_result()
    }

    pub(crate) unsafe fn set_attribute<T: ?Sized>(
        &mut self,
        aname: cudnn_sys::cudnnBackendAttributeName_t,
        atype: cudnn_sys::cudnnBackendAttributeType_t,
        count: i64,
        val: &T,
    ) -> Result<(), CudnnError> {
        let ptr = val as *const T as *const std::ffi::c_void;

        cudnn_sys::cudnnBackendSetAttribute(self.0.raw, aname, atype, count, ptr).into_result()
    }

    pub(crate) unsafe fn get_attribute_count(
        &self,
        aname: cudnn_sys::cudnnBackendAttributeName_t,
        atype: cudnn_sys::cudnnBackendAttributeType_t,
    ) -> Result<i64, CudnnError> {
        let mut count = MaybeUninit::<i64>::uninit();

        cudnn_sys::cudnnBackendGetAttribute(
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

    pub(crate) fn inner(&self) -> cudnn_sys::cudnnBackendDescriptor_t {
        self.0.raw
    }
}
