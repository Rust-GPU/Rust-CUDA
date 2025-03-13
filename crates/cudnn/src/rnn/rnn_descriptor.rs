use crate::{
    sys, CudnnError, DataType, DropoutDescriptor, IntoResult, MathType, NanPropagation, RnnAlgo,
    RnnBiasMode, RnnClipMode, RnnDirectionMode, RnnInputMode, RnnMode,
};
use cust::memory::GpuBuffer;
use std::{marker::PhantomData, mem::MaybeUninit};

bitflags::bitflags! {
    /// Miscellaneous switches for configuring auxiliary recurrent neural network features.
    pub struct RnnAuxFlags: u32 {
        /// When the padded I/O is disabled, layouts `SeqMajorUnpacked` and `BatchMajorUnpacked`
        /// are not permitted in RNN data descriptors.
        const PADDED_IO_DISABLED = 0;
        /// When the padded I/O is enabled, layouts `SeqMajorUnpacked` and `BatchMajorUnpacked`
        /// are permitted in RNN data descriptors.
        const PADDED_IO_ENABLED = 1;
    }
}

/// A description of an recurrent neural network operation.
///
/// This descriptor is generic over the data type of the parameters and the inputs, and the one of
/// the computation.
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct RnnDescriptor<T, U>
where
    T: DataType,
    U: SupportedRnn<T>,
{
    pub(crate) raw: sys::cudnnRNNDescriptor_t,
    data_type: PhantomData<T>,
    math_prec: PhantomData<U>,
}

impl<T, U> RnnDescriptor<T, U>
where
    T: DataType,
    U: SupportedRnn<T>,
{
    /// Initializes a RNN descriptor object.
    ///
    /// # Arguments
    ///
    ///   * `algo` - a recurrent neural network algorithm.
    ///   * `cell_mode` - specifies the RNN cell type in the entire model.
    ///   * `bias_mode` - number of bias vectors.
    ///   * `dir_mode` - recurrence pattern.
    ///   * `input_mode` - specifies how the input to the RNN model is processed by the
    ///     first layer. When `input_mode` is `RnnInputMode::LinearInput`, the original
    ///     input vectors of size `input_size` are multiplied by the weight matrix to
    ///     obtain vectors of hiddenSize. When `input_mode` is set to
    ///     `RnnInputMode::SkipInput`, the original input vectors to the first layer are
    ///     used as is without multiplying them by the weight matrix.
    ///   * `math_type` - preferred option to use NVIDIA Tensor Cores accelerators on
    ///     Volta (SM 7.0) or higher GPUs.
    ///   * `input_size` - size of the input vector in the RNN model.
    ///   * `hidden_size` - size of the hidden state vector in the RNN model. The same
    ///     hidden size is used in all RNN layers.
    ///   * `projection_size` - the size of the LSTM cell output after the recurrent
    ///     projection. This value should not be larger than `hidden_size`. It is legal
    ///     to set it equal to `hidden_size`, however, in this case, the recurrent
    ///     projection feature is disabled. The recurrent projection is an additional
    ///     matrix multiplication in the LSTM cell to project hidden state vectors ht
    ///     into smaller vectors rt = Wr * ht, where Wr is a rectangular matrix with
    ///     `projection_size` rows and `hidden_size` columns. When the recurrent
    ///     projection is enabled, the output of the LSTM cell (both to the next layer
    ///     and unrolled in-time) is rt instead of ht. The recurrent projection can be
    ///     enabled for LSTM cells and `RnnAlgo::AlgoStandard` only.
    ///   * `num_layers` - number of stacked, physical layers in the deep RNN model.
    ///     When `dir_mode` is equal to `RnnDirectionMode::Bidirectional`, the physical
    ///     layer consists of two pseudo-layers corresponding to forward and backward
    ///     directions.
    ///   * `dropout_desc` - an optional dropout descriptor. Dropout operation will be
    ///     applied between physical layers. A single layer network will have no dropout
    ///     applied. Dropout is used in the training mode only.
    ///   * `aux_flags` - this argument is used to pass miscellaneous switches that do
    ///     not require additional numerical values to configure the corresponding
    ///     feature. In future cuDNN releases, this parameter will be used to extend the
    ///     RNN functionality without adding new API functions (applicable options
    ///     should be bitwise OR-ed). Currently, this parameter is used to enable or
    ///     disable padded input/output (`RnnAuxFlags::PADDED_IO_DISABLED`,
    ///     `RnnAuxFlags::PADDED_IO_ENABLED`). When the padded I/O is enabled, layouts
    ///     `SeqMajorUnpacked` and `BatchMajorUnpacked` are permitted in RNN data
    ///     descriptors.
    ///
    /// cuDNN
    /// [docs](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnSetRNNDescriptor_v8)
    /// may offer additional information about the APi behavior.
    ///
    /// # Errors
    ///
    /// Returns errors if an incompatible or unsupported combination of input arguments
    /// was detected.
    ///
    /// # Examples
    ///
    /// ```
    /// # use std::error::Error;
    /// #
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// use cudnn::{
    ///     CudnnContext, DropoutDescriptor, MathType, RnnAlgo, RnnAuxFlags, RnnBiasMode,
    ///     RnnDescriptor, RnnDirectionMode, RnnInputMode, RnnMode,
    /// };
    /// use cust::memory::DeviceBuffer;
    ///
    /// let ctx = CudnnContext::new()?;
    ///
    /// let algo = RnnAlgo::Standard;
    /// let cell_mode = RnnMode::Lstm;
    /// let bias_mode = RnnBiasMode::SingleRecurrentBias;
    /// let dir_mode = RnnDirectionMode::Unidirectional;
    /// let input_mode = RnnInputMode::SkipInput;
    /// let math_type = MathType::TensorOpAllowConversion;
    /// let input_size = 20;
    /// let hidden_size = 25;
    /// let projection_size = 10;
    /// let num_layers = 3;
    /// let dropout_desc: Option<&DropoutDescriptor<DeviceBuffer<u8>>> = None;
    /// let aux_flags = RnnAuxFlags::PADDED_IO_ENABLED;
    ///
    /// let rnn_desc = RnnDescriptor::<f32, f32>::new(
    ///     algo,
    ///     cell_mode,
    ///     bias_mode,
    ///     dir_mode,
    ///     input_mode,
    ///     math_type,
    ///     input_size,
    ///     hidden_size,
    ///     projection_size,
    ///     num_layers,
    ///     dropout_desc,
    ///     aux_flags,
    /// )?;
    /// # Ok(())
    /// # }
    /// ```
    #[allow(clippy::too_many_arguments)]
    pub fn new<S>(
        algo: RnnAlgo,
        cell_mode: RnnMode,
        bias_mode: RnnBiasMode,
        dir_mode: RnnDirectionMode,
        input_mode: RnnInputMode,
        math_type: MathType,
        input_size: i32,
        hidden_size: i32,
        projection_size: impl Into<Option<i32>>,
        num_layers: i32,
        dropout_desc: Option<&DropoutDescriptor<S>>,
        aux_flags: RnnAuxFlags,
    ) -> Result<Self, CudnnError>
    where
        S: GpuBuffer<u8>,
    {
        let mut raw = MaybeUninit::uninit();

        unsafe {
            sys::cudnnCreateRNNDescriptor(raw.as_mut_ptr()).into_result()?;

            let raw = raw.assume_init();

            let proj_size = projection_size.into().unwrap_or(0);
            let dropout_desc = dropout_desc.map_or(std::ptr::null_mut(), |desc| desc.raw);

            sys::cudnnSetRNNDescriptor_v8(
                raw,
                algo.into(),
                cell_mode.into(),
                bias_mode.into(),
                dir_mode.into(),
                input_mode.into(),
                T::into_raw(),
                U::into_raw(),
                math_type.into(),
                input_size,
                hidden_size,
                proj_size,
                num_layers,
                dropout_desc,
                aux_flags.bits(),
            )
            .into_result()?;

            Ok(Self {
                raw,
                data_type: PhantomData,
                math_prec: PhantomData,
            })
        }
    }

    /// Sets the LSTM cell clipping mode. The LSTM clipping is disabled by default. When
    /// enabled, clipping is applied to all layers. This function does not affect the
    /// work, reserve, and weight-space buffer sizes and may be called multiple times.
    ///
    /// # Arguments
    ///
    ///   * `clip_mode` - enables or disables the LSTM cell clipping. When `clip_mode`
    ///     is set to `RnnClipMode::ClipNone` no LSTM cell state clipping is performed.
    ///     When `clip_mode` is `RnnClipMode::ClipMinMax` the cell state activation to
    ///     other units is clipped.
    ///
    ///  * `nan_opt` - when set to `NanPropagation::PropagateNan`, NaN is propagated
    ///    from the LSTM cell, or it can be set to one of the clipping range boundary
    ///    values, instead of propagating.
    ///   * `left_clip` - left bound of the clipping range.
    ///   * `right_clip` - right bound of the clipping range.
    ///
    /// **Do note** that cell clipping is only available if the cell mode associated to
    /// this descriptor is `RnnMode::Lstm`.
    ///
    /// cuDNN
    /// [docs](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnRNNSetClip_v8)
    /// may offer additional information about the APi behavior.
    ///
    /// # Errors
    ///
    /// Returns errors if either `left_clip` or `right_clip` is NaN or if `right_clip` <
    /// `left_clip` or if the associated cell mode is not `RnnMode::Lstm`.
    pub fn set_clip(
        &mut self,
        clip_mode: RnnClipMode,
        nan_opt: NanPropagation,
        left_clip: f64,
        right_clip: f64,
    ) -> Result<(), CudnnError> {
        unsafe {
            sys::cudnnRNNSetClip_v8(
                self.raw,
                clip_mode.into(),
                nan_opt.into(),
                left_clip,
                right_clip,
            )
            .into_result()
        }
    }
}

impl<T, U> Drop for RnnDescriptor<T, U>
where
    T: DataType,
    U: SupportedRnn<T>,
{
    fn drop(&mut self) {
        unsafe {
            sys::cudnnDestroyRNNDescriptor(self.raw);
        }
    }
}

/// Controls the compute math precision in the recurrent neural network model. The following
/// applies:
///
/// * For input and output in `f32`, the math precision of the model can only be `f32`.
///
/// * For input and output in `f64` the math precision of the model can only be `f64`.
pub trait SupportedRnn<T>
where
    Self: DataType,
    T: DataType,
{
}

impl SupportedRnn<f32> for f32 {}
impl SupportedRnn<f64> for f64 {}
