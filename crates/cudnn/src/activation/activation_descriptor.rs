use crate::{sys, ActivationMode, CudnnError, IntoResult, NanPropagation};
use std::mem::MaybeUninit;

/// The descriptor of a neuron activation operation.
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct ActivationDescriptor {
    pub(crate) raw: sys::cudnnActivationDescriptor_t,
}

impl ActivationDescriptor {
    /// Creates a new neuron activation descriptor.
    ///
    /// # Arguments
    ///
    ///   * `mode` - activation function to compute.
    ///   * `nan_opt` - NaN propagation policy for the operation.
    ///   * `coefficient` - optional coefficient for the given function. It specifies
    ///     the clipping threshold for `ActivationMode::ClippedRelu`.
    ///
    /// cuDNN
    /// [docs](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnSetActivationDescriptor)
    /// may offer additional information about the API behavior.
    ///
    /// # Examples
    ///
    /// ```
    /// # use std::error::Error;
    /// #
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// use cudnn::{ActivationDescriptor, ActivationMode, CudnnContext, NanPropagation};
    ///
    /// let ctx = CudnnContext::new()?;
    ///
    /// let mode = ActivationMode::Swish;
    /// let nan_opt = NanPropagation::PropagateNaN;
    /// let coefficient = None;
    ///
    /// let desc = ActivationDescriptor::new(mode, nan_opt, coefficient)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(
        mode: ActivationMode,
        nan_opt: NanPropagation,
        coefficient: impl Into<Option<f64>>,
    ) -> Result<Self, CudnnError> {
        let mut raw = MaybeUninit::uninit();

        unsafe {
            sys::cudnnCreateActivationDescriptor(raw.as_mut_ptr()).into_result()?;

            let raw = raw.assume_init();

            let coefficient = coefficient.into().unwrap_or(match mode {
                ActivationMode::ClippedRelu => f64::MAX,
                _ => 1.0,
            });

            sys::cudnnSetActivationDescriptor(raw, mode.into(), nan_opt.into(), coefficient)
                .into_result()?;

            Ok(Self { raw })
        }
    }
}

impl Drop for ActivationDescriptor {
    fn drop(&mut self) {
        unsafe {
            sys::cudnnDestroyActivationDescriptor(self.raw);
        }
    }
}
