mod forward_mode;
mod rnn_algo;
mod rnn_bias_mode;
mod rnn_clip_mode;
mod rnn_data_descriptor;
mod rnn_data_layout;
mod rnn_descriptor;
mod rnn_direction_mode;
mod rnn_input_mode;
mod rnn_mode;

pub use forward_mode::*;
pub use rnn_algo::*;
pub use rnn_bias_mode::*;
pub use rnn_clip_mode::*;
pub use rnn_data_descriptor::*;
pub use rnn_data_layout::*;
pub use rnn_descriptor::*;
pub use rnn_direction_mode::*;
pub use rnn_input_mode::*;
pub use rnn_mode::*;

use crate::{sys, CudnnContext, CudnnError, IntoResult, TensorDescriptor, WGradMode};
use cust::memory::GpuBuffer;
use std::mem::MaybeUninit;

impl CudnnContext {
    /// Computes the work and reserve space buffer sizes based on the RNN network
    /// geometry stored in `rnn_desc`, designated usage (inference or training) defined
    /// by the `mode` argument, and the current RNN data dimensions are retrieved from
    /// `x_desc`.
    ///
    /// When RNN data dimensions change, this function must be called again because the
    /// RNN temporary buffer sizes are not monotonic.
    ///
    /// # Arguments
    ///
    /// * `rnn_desc` - a RNN descriptor.
    /// * `mode` - specifies whether the temporary buffers are used in inference or
    ///   training mode. The reserve-space buffer is not used during inference.
    ///   Therefore, the returned size of the reserve space buffer will be `None` when
    ///   the `mode` argument is `ForwardMode::Inference`.
    /// * `x_desc` - a RNN data descriptor.
    ///
    /// cuDNN
    /// [docs](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnGetRNNTempSpaceSizes)
    /// may offer additional information about the APi behavior.
    ///
    /// # Errors
    ///
    /// Returns an error is an incompatible or unsupported combination of input
    /// arguments was detected.
    pub fn get_rnn_temp_space_sizes<T1, T2>(
        &self,
        rnn_desc: &RnnDescriptor<T1, T2>,
        forward_mode: ForwardMode,
        x_desc: &RnnDataDescriptor<T1>,
    ) -> Result<(usize, Option<usize>), CudnnError>
    where
        T1: RnnDataType,
        T2: SupportedRnn<T1>,
    {
        let mut workspace_size = MaybeUninit::uninit();
        let mut reserve_space_size = MaybeUninit::uninit();

        unsafe {
            sys::cudnnGetRNNTempSpaceSizes(
                self.raw,
                rnn_desc.raw,
                forward_mode.into(),
                x_desc.raw,
                workspace_size.as_mut_ptr(),
                reserve_space_size.as_mut_ptr(),
            )
            .into_result()?;

            Ok((
                workspace_size.assume_init(),
                match reserve_space_size.assume_init() {
                    0 => None,
                    size => Some(size),
                },
            ))
        }
    }

    /// This function returns the required size of the weight space buffer in bytes.
    /// The weight space buffer holds all RNN weight matrices and bias vectors.
    ///
    /// # Arguments
    ///
    /// `rnn_desc` - an RNN descriptor.
    ///
    /// cuDNN [docs](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnGetRNNWeightSpaceSize)
    /// may offer additional information about the APi behavior.
    pub fn get_rnn_weight_space_size<T1, T2>(
        &self,
        rnn_desc: &RnnDescriptor<T1, T2>,
    ) -> Result<usize, CudnnError>
    where
        T1: RnnDataType,
        T2: SupportedRnn<T1>,
    {
        let mut size = MaybeUninit::uninit();

        unsafe {
            sys::cudnnGetRNNWeightSpaceSize(self.raw, rnn_desc.raw, size.as_mut_ptr())
                .into_result()?;

            Ok(size.assume_init())
        }
    }

    /// This routine computes the forward response of the recurrent neural network
    /// described by `rnn_desc` with inputs in `x`, `hx`, `cx`, and weights / biases in
    /// the `weight_space` buffer. RNN outputs are written to `y`, `hy`, and `cy`
    /// buffers.
    ///
    /// Note that internal RNN signals between time-steps and between layers are not
    /// exposed to the user.
    ///
    /// When the `forward_mode` parameter is set to `ForwardMode::Training`, this
    /// function stores intermediate data required to compute first order derivatives in
    /// the reserve space buffer. Work and reserve space buffer sizes should be computed
    /// by the `get_rnn_temp_space_sizes` function with the same `forward_mode` setting
    /// as used in the `rnn_forward()` call.
    ///
    /// The same layout type must be specified in `x_desc` and `y_desc` descriptors. The
    /// same sequence lengths must be configured in `x_desc`, `y_desc` and in the device
    /// array `device_seq_lengths`. The `rnn_forward()` function does not verify that
    /// sequence lengths stored in `device_seq_lengths` in GPU memory are the same as in
    /// `x_desc` and `y_desc` descriptors in CPU memory.
    ///
    /// Sequence length arrays from `x_desc` and `y_desc` descriptors are checked for
    /// consistency, however.
    ///
    /// # Arguments
    ///
    /// * `rnn_desc` - a RNN descriptor.
    /// * `forward_mode` - specifies inference or training mode. In the training mode,
    ///   additional data is stored in the reserve space buffer. This information is
    ///   used in the backward pass to compute derivatives.
    /// * `device_seq_lengths` - a copy of `seq_lengths` from `x_desc` or `y_desc` RNN
    ///   data descriptors. The `device_seq_lengths`  must be stored in GPU memory as it
    ///   is accessed asynchronously by GPU kernels, possibly after the `rnn_forward()`
    ///   function exists.
    /// * `x_desc` -  a descriptor corresponding to the RNN model primary input.
    /// * `x` - GPU memory associated with the RNN data descriptor `x_desc`.
    /// * `y_desc` - a RNN data descriptor. The data type, layout, maximum sequence
    ///   length, batch size, and sequence lengths array must match that of `x_desc`.
    ///   The parameter `vector_size` of `y_desc`  depends on whether LSTM projection is
    ///   enabled (only for LSTM) and whether the network is bidirectional.
    ///   Specifically: for unidirectional models, the parameter `vector_size` must
    ///   match the `hidden_size` argument passed to the RNN descriptor constructor. If
    ///   the LSTM projection is enabled, the `vector_size` must be the same as the
    ///   `projection_size` argument passed to the RNN descriptor constructor. For
    ///   bidirectional models, if the RNN `cell_mode` is `RnnMode::Lstm` and the
    ///   projection feature is enabled, the parameter `vector_size` must be double the
    ///   `projection_size` argument passed to the RNN descriptor constructor.
    ///   Otherwise, it should be double the `hidden_size` value.
    /// * `y` - GPU memory associated with the RNN data descriptor `y_desc`.
    /// * `h_desc` - tensor descriptor describing the initial or final hidden state of
    ///   RNN. The first dimension of the tensor depends on the `dir_mode` argument
    ///   passed to the descriptor constructor function. If `dir_mode` is
    ///   `RnnDirectionMode::Unidirectional`, then the first dimension should match the
    ///   `num_layers` argument passed to the RNN descriptor constructor. If `dir_mode`
    ///   is `RnnDirectionMode::Bidirectional`, then the first dimension should match
    ///   double the `num_layers` argument passed to the RNN descriptor constructor. The
    ///   second dimension must match the `bath_size` parameter described in `x_desc`.
    ///   The third dimension depends on whether RNN mode is `RnnMode::Lstm` and whether
    ///   the LSTM projection is enabled. Specifically: if RNN mode is `RnnMode::Lstm`
    ///   and LSTM projection is enabled, the third dimension must match the
    ///   `projection_size` argument passed to the RNN descriptor constructor.
    ///   Otherwise, the third dimension must match the `hidden_size` argument passed to
    ///   the RNN descriptor constructor.
    /// * `hx` - GPU buffer with the RNN initial hidden state. Data dimensions are
    ///   described by the `h_desc` tensor descriptor. If `None` is passed, the initial
    ///   hidden state of the network will be initialized to zero.
    /// * `hy` - GPU buffer where the final RNN hidden state should be stored. Data
    ///   dimensions are described by the `h_desc` tensor descriptor. If `None` is
    ///   passed, the final hidden state of the network will not be saved.
    /// * `c_desc` - for LSTM networks only. A tensor descriptor describing the initial
    ///   or final cell state for LSTM networks only. The first dimension of the tensor
    ///   depends on the `dir_mode` argument passed to the RNN descriptor constructor
    ///   call.
    /// * `cx` -  For LSTM networks only. GPU buffer with the initial LSTM state data.
    ///   Data dimensions are described by the `c_desc` tensor descriptor. If `None` is
    ///   passed, the initial cell state of the network will be initialized to zero.
    /// * `cy` - For LSTM networks only. GPU buffer where final LSTM state data should
    ///   be stored. Data dimensions are described by the `c_desc` tensor descriptor. If
    ///   `None` is passed, the final LSTM cell state will not be saved.
    /// * `weight_space` - weight space buffer in GPU memory.
    /// * `work_space` - buffer in GPU memory to store temporary data.
    /// * `reserve_space` - reserve-space buffer in GPU memory.
    ///
    /// cuDNN
    /// [docs](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnRNNForward)
    /// may offer additional information about the APi behavior.
    ///
    /// # Errors
    ///
    /// Returns errors is an unsupported arguments combination is detected or if the
    /// supplied buffers are too small.
    #[allow(clippy::too_many_arguments)]
    pub fn rnn_forward<T1, T2>(
        &self,
        rnn_desc: &RnnDescriptor<T1, T2>,
        forward_mode: ForwardMode,
        device_seq_lengths: &impl GpuBuffer<i32>,
        x_desc: &RnnDataDescriptor<T1>,
        x: &impl GpuBuffer<T1>,
        y_desc: &RnnDataDescriptor<T1>,
        y: &impl GpuBuffer<T1>,
        h_desc: &TensorDescriptor<T1>,
        hx: Option<&impl GpuBuffer<T1>>,
        hy: Option<&mut impl GpuBuffer<T1>>,
        c_desc: Option<&TensorDescriptor<T1>>,
        cx: Option<&impl GpuBuffer<T1>>,
        cy: Option<&mut impl GpuBuffer<T1>>,
        weight_space: &mut impl GpuBuffer<u8>,
        work_space: &mut impl GpuBuffer<u8>,
        reserve_space: Option<&mut impl GpuBuffer<u8>>,
    ) -> Result<(), CudnnError>
    where
        T1: RnnDataType,
        T2: SupportedRnn<T1>,
    {
        let device_sequence_lengths_ptr = device_seq_lengths.as_device_ptr().as_ptr();

        let x_ptr = x.as_device_ptr().as_ptr() as *const std::ffi::c_void;
        let y_ptr = y.as_device_ptr().as_mut_ptr() as *mut std::ffi::c_void;

        let hx_ptr = hx.map_or(std::ptr::null(), |buff| {
            buff.as_device_ptr().as_ptr() as *const std::ffi::c_void
        });
        let hy_ptr = hy.map_or(std::ptr::null_mut(), |buff| {
            buff.as_device_ptr().as_mut_ptr() as *mut std::ffi::c_void
        });

        let c_desc = c_desc.map_or(std::ptr::null_mut(), |desc| desc.raw);
        let cx_ptr = cx.map_or(std::ptr::null(), |buff| {
            buff.as_device_ptr().as_ptr() as *const std::ffi::c_void
        });
        let cy_ptr = cy.map_or(std::ptr::null_mut(), |buff| {
            buff.as_device_ptr().as_mut_ptr() as *mut std::ffi::c_void
        });

        let weight_space_ptr = weight_space.as_device_ptr().as_ptr() as *const std::ffi::c_void;
        let work_space_ptr = work_space.as_device_ptr().as_mut_ptr() as *mut std::ffi::c_void;
        let (reserve_space_ptr, reserve_space_size) =
            reserve_space.map_or((std::ptr::null_mut(), 0), |buff| {
                (
                    buff.as_device_ptr().as_mut_ptr() as *mut std::ffi::c_void,
                    buff.len(),
                )
            });

        unsafe {
            sys::cudnnRNNForward(
                self.raw,
                rnn_desc.raw,
                forward_mode.into(),
                device_sequence_lengths_ptr,
                x_desc.raw,
                x_ptr,
                y_desc.raw,
                y_ptr,
                h_desc.raw,
                hx_ptr,
                hy_ptr,
                c_desc,
                cx_ptr,
                cy_ptr,
                weight_space.len(),
                weight_space_ptr,
                work_space.len(),
                work_space_ptr,
                reserve_space_size,
                reserve_space_ptr,
            )
            .into_result()
        }
    }

    /// This function computes exact, first-order derivatives of the RNN model with
    /// respect to its inputs: `x`, `hx` and for the LSTM cell type also `cx`.
    ///
    /// The following buffers should contain to the same data as in the preceding
    /// `rnn_forward()` call: `y`, the initial hidden state `hx`, and the initial cell
    /// state `cx` (for LSTM only).
    ///
    /// This function accepts any combination of `dhy`, `dhx`, `dcy`, `dcx` being
    /// `None`; when `dhy` or `dcy` are `None`, it is assumed that those inputs are
    /// zero. When `dhx` or `dcx` are `None` then the corresponding results are not
    /// written by this function.
    ///
    /// When all `hx`, `dhy`, `dhx` are `None`, then the corresponding tensor descriptor
    /// `h_desc` can be `None` too. The same rule applies to the `cx`, `dcy`, `dcx`
    /// pointers and the `c_desc` tensor descriptor.
    ///
    /// This function allows the user to use padded layouts for inputs `y`, `dy`, and
    /// output `dx`. In padded or unpacked layouts (`RnnDataLayout::SeqMajorUnpacked`,
    /// `RnnDataLayout::BatchMajorUnpacked`) each sequence of vectors in a mini-batch
    /// has a fixed length defined by the `max_seq_length` argument in the
    /// `RnnDataDescriptor` constructor function. The term *"unpacked"* refers here to
    /// the presence of padding vectors. Each padded, fixed-length sequence starts from
    /// a segment of valid vectors. The valid vector count is stored in `seq_lengths`,
    /// such that 0 < `seq_lengths[i]` <= `max_seq_length` for all sequences in a
    /// mini-batch, i.e., for i in 0..`batch_size` - 1. The remaining padding vectors
    /// make the combined sequence length equal to `max_seq_length`. Both sequence-major
    /// and batch-major padded layouts are supported.
    ///
    /// In addition, a packed sequence-major layout: `RnnDataLayout::SeqMajorPacked` can
    /// be selected by the user. In the latter layout, sequences of vectors in a
    /// mini-batch are sorted in the descending order according to the sequence lengths.
    /// First, all vectors for time step zero are stored. They are followed by vectors
    /// for time step one, and so on. This layout uses no padding vectors.
    ///
    /// **Do note** that the same layout type must be specified in `x_desc` and `y_desc`
    /// descriptors.
    ///
    /// Two host arrays named `seq_lengths` in `x_desc` and `y_desc` RNN data
    /// descriptors must be the same. In addition, a copy of `seq_lengths` in the device
    /// memory must be passed via the `device_seq_lengths` argument. This array is
    /// supplied directly to GPU kernels. This function does not verify that sequence
    /// lengths stored in `device_seq_lengths` in GPU memory are the same as in `x_desc`
    /// and `y_desc` descriptors in CPU memory.
    ///
    /// Sequence length arrays from `x_desc` and `y_desc` descriptors are checked for
    /// consistency, however.
    ///
    /// This function must be called after `rnn_forward()`. The latter function should
    /// be invoked with the `forward_mode` argument of type `ForwardMode::Training`.
    ///
    /// # Arguments
    ///
    /// * `rnn_desc` - RNN descriptor.
    /// * `device_seq_lengths` -  a copy of `seq_lengths` from `x_desc` or `y_desc` RNN
    ///   data descriptors. The `device_seq_lengths` array must be stored in GPU memory
    ///   as it is accessed asynchronously by GPU kernels, possibly after this function
    ///   exists.
    /// * `y_desc` - previously initialized descriptor corresponding to the RNN model
    ///   primary output. The data type, layout, `max_seq_length`, `batch_size`, and
    ///   `seq_lengths` need to match that of `x_desc`.
    /// * `y` - GPU buffer holding the model's primary output.
    /// * `dy` - GPU buffer holding the gradient of the loss w.r.t. `y`. The `y` output
    ///   should be produced by the preceding `rnn_forward()` call. The `y` and `dy`
    ///   vectors are expected to be laid out in memory according to the layout
    ///   specified by `y_desc`.
    /// * `x_desc` - RNN data descriptor corresponding to the gradient of the loss
    ///   function with respect to the RNN primary model input.
    /// * `x` -  GPU buffer where back-propagated gradients of the loss function with
    ///   respect to the RNN primary input x should be stored.
    /// * `h_desc` - tensor descriptor describing the initial or final hidden state of
    ///   RNN. The first dimension of the tensor depends on the `dir_mode` argument
    ///   passed to the descriptor constructor function. If `dir_mode` is
    ///   `RnnDirectionMode::Unidirectional`, then the first dimension should match the
    ///   `num_layers` argument passed to the RNN descriptor constructor. If `dir_mode`
    ///   is `RnnDirectionMode::Bidirectional`, then the first dimension should match
    ///   double the `num_layers` argument passed to the RNN descriptor constructor. The
    ///   second dimension must match the `bath_size` parameter described in `x_desc`.
    ///   The third dimension depends on whether RNN mode is `RnnMode::Lstm` and whether
    ///   the LSTM projection is enabled. Specifically: if RNN mode is `RnnMode::Lstm`
    ///   and LSTM projection is enabled, the third dimension must match the
    ///   `projection_size` argument passed to the RNN descriptor constructor.
    ///   Otherwise, the third dimension must match the `hidden_size` argument passed to
    ///   the RNN descriptor constructor.
    /// * `hx` - GPU buffer with the RNN initial hidden state. Data dimensions are
    ///   described by the `h_desc` tensor descriptor. If `None` is passed in `hx` the
    ///   corresponding buffer is assumed to contain all zeros.
    /// * `dhy` - GPU buffer with the RNN gradient deltas for the hidden state. Data
    ///   dimensions are described by the `h_desc` tensor descriptor. If `None` is
    ///   passed in `dhy` the corresponding buffer is assumed to contain all zeros.
    /// * `dhx` -  GPU buffer where first-order derivatives corresponding to initial
    ///   hidden state variables should be stored. Data dimensions are described by the
    ///   `h_desc` tensor descriptor. If `None`  is passed as `dhx`, the back-propagated
    ///   derivatives are not saved.
    /// * `c_desc` - for LSTM networks only. A tensor descriptor describing the initial
    ///   or final cell state for LSTM networks only. The first dimension of the tensor
    ///   depends on the `dir_mode` argument passed to the RNN descriptor constructor
    ///   call.
    /// * `cx` -  For LSTM networks only. GPU buffer with the initial LSTM state data.
    ///   Data dimensions are described by the `c_desc` tensor descriptor. If `None` is
    ///   passed, the initial cell state of the network is assumed to contain all zeros.
    /// * `dcy` -  For LSTM networks only. GPU buffer with the gradient deltas for the
    ///   LSTM state. Data  dimensions are described by the `c_desc` tensor descriptor.
    ///   If `None` is passed, the buffer is assumed to contain all zeros.
    /// * `dcx` -  For LSTM networks only. GPU buffer where first-order derivatives
    ///   corresponding to initial LSTM state variables should be stored. Data
    ///   dimensions are described by the `c_desc` tensor descriptor. If `None` is
    ///   assigned to `dcx`, the back-propagated derivatives are not saved.
    /// * `weight_space` - weight space buffer in GPU memory.
    /// * `work_space` - workspace buffer in GPU memory to store temporary data.
    /// * `reserve_space` - reserve-space buffer in GPU memory.
    ///
    /// cuDNN
    /// [docs](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnRNNBackwardData_v8)
    /// may offer additional information about the APi behavior.
    ///
    /// # Errors
    ///
    /// Returns errors if an invalid or incompatible input argument was encountered.
    #[allow(clippy::too_many_arguments)]
    pub fn rnn_backward_data<T1, T2>(
        &self,
        rnn_desc: &RnnDescriptor<T1, T2>,
        device_seq_lengths: &impl GpuBuffer<i32>,
        y_desc: &RnnDataDescriptor<T1>,
        y: &impl GpuBuffer<T1>,
        dy: &impl GpuBuffer<T1>,
        x_desc: &RnnDataDescriptor<T1>,
        dx: &mut impl GpuBuffer<T1>,
        h_desc: Option<&TensorDescriptor<T1>>,
        hx: Option<&impl GpuBuffer<T1>>,
        dhy: Option<&impl GpuBuffer<T1>>,
        dhx: Option<&mut impl GpuBuffer<T1>>,
        c_desc: Option<&TensorDescriptor<T1>>,
        cx: Option<&impl GpuBuffer<T1>>,
        dcy: Option<&impl GpuBuffer<T1>>,
        dcx: Option<&mut impl GpuBuffer<T1>>,
        weight_space: &mut impl GpuBuffer<u8>,
        work_space: &mut impl GpuBuffer<u8>,
        reserve_space: &mut impl GpuBuffer<u8>,
    ) -> Result<(), CudnnError>
    where
        T1: RnnDataType,
        T2: SupportedRnn<T1>,
    {
        let device_sequence_lengths_ptr = device_seq_lengths.as_device_ptr().as_ptr();

        let y_ptr = y.as_device_ptr().as_ptr() as *const std::ffi::c_void;
        let dy_ptr = dy.as_device_ptr().as_ptr() as *const std::ffi::c_void;

        let dx_ptr = dx.as_device_ptr().as_mut_ptr() as *mut std::ffi::c_void;

        let h_desc = h_desc.map_or(std::ptr::null_mut(), |desc| desc.raw);

        let hx_ptr = hx.map_or(std::ptr::null(), |buff| {
            buff.as_device_ptr().as_ptr() as *const std::ffi::c_void
        });
        let dhy_ptr = dhy.map_or(std::ptr::null(), |buff| {
            buff.as_device_ptr().as_ptr() as *const std::ffi::c_void
        });
        let dhx_ptr = dhx.map_or(std::ptr::null_mut(), |buff| {
            buff.as_device_ptr().as_mut_ptr() as *mut std::ffi::c_void
        });

        let c_desc = c_desc.map_or(std::ptr::null_mut(), |desc| desc.raw);

        let cx_ptr = cx.map_or(std::ptr::null(), |buff| {
            buff.as_device_ptr().as_ptr() as *const std::ffi::c_void
        });
        let dcy_ptr = dcy.map_or(std::ptr::null(), |buff| {
            buff.as_device_ptr().as_ptr() as *const std::ffi::c_void
        });
        let dcx_ptr = dcx.map_or(std::ptr::null_mut(), |buff| {
            buff.as_device_ptr().as_mut_ptr() as *mut std::ffi::c_void
        });

        let weight_space_ptr = weight_space.as_device_ptr().as_mut_ptr() as *mut std::ffi::c_void;
        let work_space_ptr = work_space.as_device_ptr().as_mut_ptr() as *mut std::ffi::c_void;
        let reserve_space_ptr = reserve_space.as_device_ptr().as_mut_ptr() as *mut std::ffi::c_void;

        unsafe {
            sys::cudnnRNNBackwardData_v8(
                self.raw,
                rnn_desc.raw,
                device_sequence_lengths_ptr,
                y_desc.raw,
                y_ptr,
                dy_ptr,
                x_desc.raw,
                dx_ptr,
                h_desc,
                hx_ptr,
                dhy_ptr,
                dhx_ptr,
                c_desc,
                cx_ptr,
                dcy_ptr,
                dcx_ptr,
                weight_space.len(),
                weight_space_ptr,
                work_space.len(),
                work_space_ptr,
                reserve_space.len(),
                reserve_space_ptr,
            )
            .into_result()
        }
    }

    /// This function computes exact, first-order derivatives of the RNN model with
    /// respect to all trainable parameters: weights and biases.
    ///
    /// This function should ne called after `rnn_forward()`.
    ///
    /// Gradient of the loss function with respect to weights and biases is typically
    /// computed over multiple mini-batches. In such a case, partial results computed
    /// for each mini-batch should be aggregated. The `add_grad` argument specifies if
    /// gradients from the current mini-batch should be added to previously computed
    /// results (`WGradMode::Add`) or the `dweight_space` buffer should be overwritten
    /// with the new results (`WGradMode::Set`).
    ///
    /// All gradient results with respect to weights and biases are written to the
    /// `d_weight_space` buffer. The size and the organization of the `dweight_space`
    /// buffer is the same as the `weight_space` buffer that holds RNN weights and
    /// biases.
    ///
    /// **Do note that** currently, this function supports the `WGradMode::Add` mode
    /// only so the `dweight_space` buffer should be zeroed by the user before invoking
    /// the routine for the first time. The same sequence lengths must be specified in
    /// the `x_desc` descriptor and in the device array `device_seq_lengths`.
    ///
    /// # Arguments
    ///
    ///   * `rnn_desc` - RNN descriptor.
    ///   * `grad_mode` -  weight gradient output mode. Only `WGradMode::Add` is
    ///     supported.
    ///   * `device_seq_lengths` -  a copy of `seq_lengths` from `x_desc` or `y_desc`
    ///     RNN data descriptors. The `device_seq_lengths` array must be stored in GPU
    ///     memory as it is accessed asynchronously by GPU kernels, possibly after this
    ///     function exists.
    ///   * `x_desc` -  a descriptor corresponding to the RNN model primary input.
    ///   * `x` - GPU memory associated with the RNN data descriptor `x_desc`.
    ///   * `h_desc` - a tensor descriptor describing the initial RNN hidden state. This
    ///     is the same tensor descriptor as used in prior `rnn_forward()` and
    ///     `rnn_backward_data()` calls.
    ///   * `hx` -  GPU buffer with the RNN initial hidden state. The same buffer `hx`
    ///     should be provided in prior `rnn_forward()` and `rnn_backward_data()` calls.
    ///   * `y_desc` - previously initialized descriptor corresponding to the RNN model
    ///     primary output. The data type, layout, `max_seq_length`, `batch_size`, and
    ///     `seq_lengths` need to match that of `x_desc`. This is the same RNN data
    ///     descriptor as used in prior `rnn_forward()` and `rnn_backward_data()` calls.
    ///   * `y` - GPU buffer holding the model's primary output.
    ///   * `dweight_space` - weights gradient space buffer in GPU memory.
    ///   * `work_space` - workspace buffer in GPU memory to store temporary data.
    ///   * `reserve_space` - reserve-space buffer in GPU memory.
    ///
    /// cuDNN
    /// [docs](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnRNNBackwardWeights_v8)
    /// may offer additional information about the APi behavior.
    ///
    /// # Errors
    ///
    /// Returns errors if an invalid or incompatible input argument combinations was
    /// encountered.
    #[allow(clippy::too_many_arguments)]
    pub fn rnn_backward_weights<T1, T2>(
        &self,
        rnn_desc: &RnnDescriptor<T1, T2>,
        grad_mode: WGradMode,
        device_seq_lengths: &impl GpuBuffer<i32>,
        x_desc: &RnnDataDescriptor<T1>,
        x: &impl GpuBuffer<T1>,
        h_desc: &TensorDescriptor<T1>,
        hx: &impl GpuBuffer<T1>,
        y_desc: &RnnDataDescriptor<T1>,
        y: &impl GpuBuffer<T1>,
        dweight_space: &mut impl GpuBuffer<u8>,
        work_space: &mut impl GpuBuffer<u8>,
        reserve_space: &mut impl GpuBuffer<u8>,
    ) -> Result<(), CudnnError>
    where
        T1: RnnDataType,
        T2: SupportedRnn<T1>,
    {
        let device_sequence_lengths_ptr = device_seq_lengths.as_device_ptr().as_mut_ptr();

        let x_ptr = x.as_device_ptr().as_ptr() as *const std::ffi::c_void;
        let hx_ptr = hx.as_device_ptr().as_ptr() as *const std::ffi::c_void;
        let y_ptr = y.as_device_ptr().as_ptr() as *const std::ffi::c_void;

        let dweight_space_ptr = dweight_space.as_device_ptr().as_mut_ptr() as *mut std::ffi::c_void;
        let work_space_ptr = work_space.as_device_ptr().as_mut_ptr() as *mut std::ffi::c_void;
        let reserve_space_ptr = reserve_space.as_device_ptr().as_mut_ptr() as *mut std::ffi::c_void;

        unsafe {
            sys::cudnnRNNBackwardWeights_v8(
                self.raw,
                rnn_desc.raw,
                grad_mode.into(),
                device_sequence_lengths_ptr,
                x_desc.raw,
                x_ptr,
                h_desc.raw,
                hx_ptr,
                y_desc.raw,
                y_ptr,
                work_space.len(),
                work_space_ptr,
                dweight_space.len(),
                dweight_space_ptr,
                reserve_space.len(),
                reserve_space_ptr,
            )
            .into_result()
        }
    }
}
