use crate::{sys, CudnnError, IntoResult, NanPropagation, PoolingMode};
use std::mem::MaybeUninit;

/// The descriptor of a pooling operation.
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct PoolingDescriptor {
    pub(crate) raw: sys::cudnnPoolingDescriptor_t,
}

impl PoolingDescriptor {
    /// Creates a new pooling descriptor.
    ///
    /// # Arguments
    ///
    /// * `mode` - pooling mode.
    /// * `nan_opt` - nan propagation policy.
    /// * `window_shape` - shape of the pooling window.
    /// * `padding` - padding size for each dimension. Negative padding is allowed.
    /// * `stride` - stride for each dimension.
    ///
    /// cuDNN [docs](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnSetPoolingNdDescriptor)
    /// may offer additional information about the APi behavior.
    ///
    /// # Errors
    ///
    /// Returns errors if an invalid configuration of arguments is detected.
    ///
    /// # Examples
    ///
    /// ```
    /// # use std::error::Error;
    /// #
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// use cudnn::{CudnnContext, NanPropagation, PoolingDescriptor, PoolingMode};
    ///
    /// let ctx = CudnnContext::new()?;
    ///
    /// let mode = PoolingMode::Max;
    /// let nan_opt = NanPropagation::PropagateNaN;
    /// let window_shape = [2, 2];
    /// let padding = [0, 0];
    /// let stride = [1, 1];
    ///
    /// let pooling_desc = PoolingDescriptor::new(mode, nan_opt, window_shape, padding, stride)?;
    ///
    /// # Ok(())
    /// # }
    /// ```
    pub fn new<const N: usize>(
        mode: PoolingMode,
        nan_opt: NanPropagation,
        window_shape: [i32; N],
        padding: [i32; N],
        stride: [i32; N],
    ) -> Result<Self, CudnnError> {
        let mut raw = MaybeUninit::uninit();

        unsafe {
            sys::cudnnCreatePoolingDescriptor(raw.as_mut_ptr()).into_result()?;

            let raw = raw.assume_init();

            sys::cudnnSetPoolingNdDescriptor(
                raw,
                mode.into(),
                nan_opt.into(),
                N as i32,
                window_shape.as_ptr(),
                padding.as_ptr(),
                stride.as_ptr(),
            )
            .into_result()?;

            Ok(Self { raw })
        }
    }
}

impl Drop for PoolingDescriptor {
    fn drop(&mut self) {
        unsafe {
            sys::cudnnDestroyPoolingDescriptor(self.raw);
        }
    }
}
