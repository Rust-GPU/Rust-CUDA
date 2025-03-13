mod attention_descriptor;
mod attention_weights_kind;
mod seq_data_axis;
mod seq_data_descriptor;

pub use attention_descriptor::*;
pub use attention_weights_kind::*;
pub use seq_data_axis::*;
pub use seq_data_descriptor::*;

use crate::{sys, CudnnContext, CudnnError, IntoResult, WGradMode};
use cust::memory::GpuBuffer;
use std::mem::MaybeUninit;

impl CudnnContext {
    /// This function computes weight, work, and reserve space buffer sizes used by the
    /// following functions:
    ///
    ///   * `multi_head_attn_forward()`
    ///   * `multi_head_attn_backward_data()`
    ///   * `multi_head_attn_backward_weights()`
    ///
    /// # Arguments
    ///
    /// `desc` - multi-head attention descriptor.
    ///
    /// cuDNN
    /// [docs](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnGetMultiHeadAttnBuffers)
    /// may offer additional information about the APi behavior.
    ///
    /// # Errors
    ///
    /// Returns errors if invalid arguments are detected.
    pub fn get_attn_buffers_size<T, U, D1, D2>(
        &self,
        desc: &AttentionDescriptor<T, U, D1, D2>,
    ) -> Result<(usize, usize, usize), CudnnError>
    where
        T: SeqDataType,
        U: SupportedAttn<T>,
        D1: GpuBuffer<u8>,
        D2: GpuBuffer<u8>,
    {
        let mut weight_space_size = MaybeUninit::uninit();
        let mut work_space_size = MaybeUninit::uninit();
        let mut reserve_space_size = MaybeUninit::uninit();

        unsafe {
            sys::cudnnGetMultiHeadAttnBuffers(
                self.raw,
                desc.raw,
                weight_space_size.as_mut_ptr(),
                work_space_size.as_mut_ptr(),
                reserve_space_size.as_mut_ptr(),
            )
            .into_result()?;

            Ok((
                weight_space_size.assume_init(),
                work_space_size.assume_init(),
                reserve_space_size.assume_init(),
            ))
        }
    }

    /// Computes the forward response of a multi-head attention layer.
    ///
    /// When `reserve_space` is `None` the function operates in the inference mode in
    /// which backward functions are not invoked, otherwise, the training mode is
    /// assumed.
    ///
    /// # Arguments
    ///
    ///   * `attn_desc` - multi-head attention descriptor.
    ///   * `current_idx` - time-step in queries to process. When the such argument is
    ///     negative, all Q time-steps are processed. When `current_idx` is zero or
    ///     positive, the forward response is computed for the selected time-step only.
    ///   * `lo_win_idx` - integer array specifying the start indices of the attention
    ///     window for each Q time-step. The start index in K, V sets is inclusive.
    ///   * `hi_win_idx` - integer array specifying the end indices of the attention
    ///     window for each Q time-step. The end index is exclusive.
    ///   * `device_seq_lengths_qo` - device array specifying sequence lengths of query,
    ///     residual, and output sequence data.
    ///   * `device_seq_lengths_kv` - device array specifying sequence lengths of key
    ///     and value input data.
    ///   * `q_desc` - descriptor for the query and residual sequence data.
    ///   * `queries` - queries data in the device memory.
    ///   * `residuals` - residual data in device memory. Set this argument to `None` if
    ///     no residual connections are required.
    ///   * `k_desc` - descriptor for the keys sequence data.
    ///   * `keys` - keys data in device memory.
    ///   * `v_desc` - descriptor for the values sequence data.
    ///   * `values` - values data in device memory.
    ///   * `o_desc` - descriptor for the out sequence data.
    ///   * `out` - out data in device memory.
    ///   * `weights` - weights buffer in device memory.
    ///   * `work_space` - work space buffer in device memory.
    ///   * `reserve_space` - reserve space buffer in device memory. This argument
    ///     should be `None` in inference mode.
    ///
    /// cuDNN
    /// [docs](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnMultiHeadAttnForward)
    /// may offer additional information about the APi behavior.
    #[allow(clippy::too_many_arguments)]
    pub fn multi_head_attn_forward<T, U, D1, D2>(
        &self,
        attn_desc: &AttentionDescriptor<T, U, D1, D2>,
        current_idx: i32,
        lo_win_idx: &[i32],
        hi_win_idx: &[i32],
        device_seq_lengths_qo: &impl GpuBuffer<i32>,
        device_seq_lengths_kv: &impl GpuBuffer<i32>,
        q_desc: &SeqDataDescriptor<T>,
        queries: &impl GpuBuffer<T>,
        residuals: Option<&impl GpuBuffer<T>>,
        k_desc: &SeqDataDescriptor<T>,
        keys: &impl GpuBuffer<T>,
        v_desc: &SeqDataDescriptor<T>,
        values: &impl GpuBuffer<T>,
        o_desc: &SeqDataDescriptor<T>,
        out: &mut impl GpuBuffer<T>,
        weights: &impl GpuBuffer<u8>,
        work_space: &mut impl GpuBuffer<u8>,
        reserve_space: Option<&mut impl GpuBuffer<u8>>,
    ) -> Result<(), CudnnError>
    where
        T: SeqDataType,
        U: SupportedAttn<T>,
        D1: GpuBuffer<u8>,
        D2: GpuBuffer<u8>,
    {
        let device_seq_lenghts_qo_ptr = device_seq_lengths_qo.as_device_ptr().as_ptr() as *const _;
        let device_seq_lengths_kv_ptr = device_seq_lengths_kv.as_device_ptr().as_ptr() as *const _;

        let queries_ptr = queries.as_device_ptr().as_ptr() as *const _;
        let residuals_ptr = residuals.map_or(std::ptr::null(), |buff| {
            buff.as_device_ptr().as_ptr() as *const _
        });
        let keys_ptr = keys.as_device_ptr().as_ptr() as *const _;
        let values_ptr = values.as_device_ptr().as_ptr() as *const _;
        let out_ptr = out.as_device_ptr().as_mut_ptr() as *mut _;

        let weights_ptr = weights.as_device_ptr().as_ptr() as *const _;
        let work_space_ptr = work_space.as_device_ptr().as_mut_ptr() as *mut _;

        let (reserve_space_ptr, reserve_space_size) = reserve_space
            .map_or((std::ptr::null_mut(), 0), |buff| {
                (buff.as_device_ptr().as_mut_ptr() as *mut _, 0)
            });

        unsafe {
            sys::cudnnMultiHeadAttnForward(
                self.raw,
                attn_desc.raw,
                current_idx,
                lo_win_idx.as_ptr(),
                hi_win_idx.as_ptr(),
                device_seq_lenghts_qo_ptr,
                device_seq_lengths_kv_ptr,
                q_desc.raw,
                queries_ptr,
                residuals_ptr,
                k_desc.raw,
                keys_ptr,
                v_desc.raw,
                values_ptr,
                o_desc.raw,
                out_ptr,
                weights.len(),
                weights_ptr,
                work_space.len(),
                work_space_ptr,
                reserve_space_size,
                reserve_space_ptr,
            )
            .into_result()
        }
    }

    /// Computes exact, first-order derivatives of the multi-head attention block with
    /// respect to its inputs: Q, K, V.
    ///
    /// This function does not output partial derivatives for residual connections
    /// because this result is equal to `d_out`. If the multi-head attention model
    /// enables residual connections sourced directly from Q, then the `d_out` tensor
    /// needs to be added to `d_queries` to obtain the correct result of the latter.
    ///
    /// This function must be invoked after `multi_head_attn_forward()`. The
    /// `lo_win_idx`, `hi_win_idx`, `queries`, `keys`, `values`, `weights`, and
    /// `reserve_space` arguments should be the same as in the
    /// `multi_head_attn_forward()` call.
    ///
    /// Furthermore, `device_seq_lengths_dqdo` and `device_seq_lengths_dkdv` device
    /// buffers should contain the same start and end attention window indices as
    /// `device_seq_lengths_qo` and `device_seq_lengths_qo` as in the forward function
    /// invocation.
    ///
    /// **Do note** that this function does not verify that sequence lengths stored in
    /// `device_seq_lengths_dqdo` and `device_seq_lengths_dkdv` contain the same
    /// settings as `seq_lengths` in the corresponding sequence data descriptor.
    ///
    /// # Arguments
    ///
    ///   * `attn_desc` - multi-head attention descriptor.
    ///   * `lo_win_idx` - integer array specifying the start indices of the attention
    ///     window for each Q time-step. The start index in K, V sets is inclusive.
    ///   * `hi_win_idx` - integer array specifying the end indices of the attention
    ///     window for each Q time-step. The end index is exclusive.
    ///   * `device_seq_lengths_dqdo` - device buffer containing a copy of the sequence
    ///     length array from the `dq_desc` or `do_desc` sequence data descriptors.
    ///   * `device_seq_lengths_dkdv` - device buffer containing a copy of the sequence
    ///     length array from the `dk_desc` or `dv_desc` sequence data descriptors.
    ///   * `do_desc` - descriptor for the output differential, i.e. the vectors of
    ///     partial derivatives of the loss function with respect to the multi-head
    ///     attention outputs.
    ///   * `d_out` - output differential.
    ///   * `dq_desc` - descriptor for the queries differential.
    ///   * `d_queries` - gradients of the loss function computed with respect to
    ///     queries vectors.
    ///   * `queries` - queries data. This must be the same input as in
    ///     `multi_head_attn_forward()`.
    ///   * `dk_desc` - descriptor for the keys and keys gradient sequence data.
    ///   * `d_keys` - gradients of the loss function computed with respect to keys
    ///     vectors.
    ///   * `keys` - keys data. This must be the same input as in
    ///     `multi_head_attn_forward()`.
    ///   * `dv_desc` - descriptor for values and values gradient sequence data.
    ///   * `d_values` - gradients of the loss function computed with respect to values
    ///     vectors.
    ///   * `values` - values data. This must be the same input as in
    ///     `multi_head_attn_forward()`.
    ///   * `weights` - weights buffer in the device memory.
    ///   * `work_space` - work space buffer in device memory. Used for temporary API
    ///     storage.
    ///   * `reserve_space` - reserve space buffer in device memory.
    ///
    /// cuDNN
    /// [docs](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnMultiHeadAttnBackwardData)
    /// may offer additional information about the APi behavior.
    ///
    /// # Errors
    ///
    /// Returns errors if an invalid or incompatible input argument was encountered, an
    /// inconsistent internal state was encountered, a requested option or a combination
    /// of input arguments is not supported or in case of insufficient amount of shared
    /// memory to launch the kernel.
    #[allow(clippy::too_many_arguments)]
    pub fn multi_head_attn_backward_data<T, U, D1, D2>(
        &self,
        attn_desc: &AttentionDescriptor<T, U, D1, D2>,
        lo_win_idx: &[i32],
        hi_win_idx: &[i32],
        device_seq_lengths_dqdo: &impl GpuBuffer<i32>,
        device_seq_lengths_dkdv: &impl GpuBuffer<i32>,
        do_desc: &SeqDataDescriptor<T>,
        d_out: &impl GpuBuffer<T>,
        dq_desc: &SeqDataDescriptor<T>,
        d_queries: &mut impl GpuBuffer<T>,
        queries: &impl GpuBuffer<T>,
        dk_desc: &SeqDataDescriptor<T>,
        d_keys: &mut impl GpuBuffer<T>,
        keys: &impl GpuBuffer<T>,
        dv_desc: &SeqDataDescriptor<T>,
        d_values: &mut impl GpuBuffer<T>,
        values: &impl GpuBuffer<T>,
        weights: &impl GpuBuffer<u8>,
        work_space: &mut impl GpuBuffer<u8>,
        reserve_space: &mut impl GpuBuffer<u8>,
    ) -> Result<(), CudnnError>
    where
        T: SeqDataType,
        U: SupportedAttn<T>,
        D1: GpuBuffer<u8>,
        D2: GpuBuffer<u8>,
    {
        let device_seq_lengths_dqdo_ptr =
            device_seq_lengths_dqdo.as_device_ptr().as_ptr() as *const _;
        let device_seq_lengths_dkdv_ptr =
            device_seq_lengths_dkdv.as_device_ptr().as_ptr() as *const _;

        let d_out_ptr = d_out.as_device_ptr().as_ptr() as *const _;

        let d_queries_ptr = d_queries.as_device_ptr().as_mut_ptr() as *mut _;
        let queries_ptr = queries.as_device_ptr().as_ptr() as *const _;

        let d_keys_ptr = d_keys.as_device_ptr().as_mut_ptr() as *mut _;
        let keys_ptr = keys.as_device_ptr().as_ptr() as *const _;

        let d_values_ptr = d_values.as_device_ptr().as_mut_ptr() as *mut _;
        let values_ptr = values.as_device_ptr().as_ptr() as *const _;

        let weights_ptr = weights.as_device_ptr().as_ptr() as *const _;
        let work_space_ptr = work_space.as_device_ptr().as_mut_ptr() as *mut _;
        let reserve_space_ptr = reserve_space.as_device_ptr().as_mut_ptr() as *mut _;

        unsafe {
            sys::cudnnMultiHeadAttnBackwardData(
                self.raw,
                attn_desc.raw,
                lo_win_idx.as_ptr(),
                hi_win_idx.as_ptr(),
                device_seq_lengths_dqdo_ptr,
                device_seq_lengths_dkdv_ptr,
                do_desc.raw,
                d_out_ptr,
                dq_desc.raw,
                d_queries_ptr,
                queries_ptr,
                dk_desc.raw,
                d_keys_ptr,
                keys_ptr,
                dv_desc.raw,
                d_values_ptr,
                values_ptr,
                weights.len(),
                weights_ptr,
                work_space.len(),
                work_space_ptr,
                reserve_space.len(),
                reserve_space_ptr,
            )
            .into_result()
        }
    }

    /// This function computes exact, first-order derivatives of the multi-head
    /// attention block with respect to its trainable parameters: projection weights and
    /// projection biases.
    ///
    /// All gradient results with respect to weights and biases are written to the
    /// `d_weights` buffer. The size and the organization of the `d_weights` buffer is
    /// the same as the `weights` buffer that holds multi-head attention weights and
    /// biases.
    ///
    /// Gradient of the loss function with respect to weights or biases is typically
    /// computed over multiple batches. In such a case, partial results computed for
    /// each batch should be summed together. The `grad_mode` argument specifies if the
    /// gradients from the current batch should be added to previously computed results
    /// or the `d_weights` buffer should be overwritten with the new results.
    ///
    /// **Do note** that this function should be invoked **after**
    /// `multi_head_attn_backward_data()`. Also, the `queries`, `keys`, `values`,
    /// `weights`, and `reserve_space` arguments should be the same as in
    /// `multi_head_attn_fwd()` and `multi_head_attn_backward_data()` calls. The `d_out`
    /// argument should be the same as in `multi_head_attn_backward_data()`.
    ///
    /// # Arguments
    ///
    ///   * `attn_desc` - multi-head attention descriptor.
    ///   * `grad_mode` - gradient accumulation mode.
    ///   * `q_desc` - descriptor for the query and residual sequence data.
    ///   * `queries` - queries data in the device memory.
    ///   * `k_desc` - descriptor for the keys sequence data.
    ///   * `keys` - keys data in device memory.
    ///   * `v_desc` - descriptor for the values sequence data.
    ///   * `values` - values data in device memory.
    ///   * `do_desc` - descriptor for the output differential sequence data.
    ///   * `d_out` - output differential data in device memory.
    ///   * `weights` - weights buffer in the device memory.
    ///   * `d_weights` - weights gradient buffer in the device memory.
    ///   * `work_space` - work space buffer in device memory. Used for temporary API
    ///     storage.
    ///   * `reserve_space` - reserve space buffer in device memory.
    ///
    /// cuDNN
    /// [docs](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnMultiHeadAttnBackwardWeights)
    /// may offer additional information about the APi behavior.
    ///
    /// # Errors
    ///
    /// Returns errors if an invalid or incompatible input argument was encountered, an
    /// inconsistent internal state was encountered, a requested option or a combination
    /// of input arguments is not supported or in case of insufficient amount of shared
    /// memory to launch the kernel.
    #[allow(clippy::too_many_arguments)]
    pub fn multi_head_attn_backward_weights<T, U, D1, D2>(
        &self,
        attn_desc: &AttentionDescriptor<T, U, D1, D2>,
        grad_mode: WGradMode,
        q_desc: &SeqDataDescriptor<T>,
        queries: &impl GpuBuffer<T>,
        k_desc: &SeqDataDescriptor<T>,
        keys: &impl GpuBuffer<T>,
        v_desc: &SeqDataDescriptor<T>,
        values: &impl GpuBuffer<T>,
        do_desc: &SeqDataDescriptor<T>,
        d_out: &impl GpuBuffer<T>,
        weights: &impl GpuBuffer<u8>,
        d_weights: &mut impl GpuBuffer<u8>,
        work_space: &mut impl GpuBuffer<u8>,
        reserve_space: &mut impl GpuBuffer<u8>,
    ) -> Result<(), CudnnError>
    where
        T: SeqDataType,
        U: SupportedAttn<T>,
        D1: GpuBuffer<u8>,
        D2: GpuBuffer<u8>,
    {
        let queries_ptr = queries.as_device_ptr().as_ptr() as *const _;
        let keys_ptr = keys.as_device_ptr().as_ptr() as *const _;
        let values_ptr = values.as_device_ptr().as_ptr() as *const _;

        let d_out_ptr = d_out.as_device_ptr().as_ptr() as *const _;

        let weights_ptr = weights.as_device_ptr().as_ptr() as *const _;
        let d_weights_ptr = d_weights.as_device_ptr().as_mut_ptr() as *mut _;
        let work_space_ptr = work_space.as_device_ptr().as_mut_ptr() as *mut _;
        let reserve_space_ptr = reserve_space.as_device_ptr().as_mut_ptr() as *mut _;

        unsafe {
            sys::cudnnMultiHeadAttnBackwardWeights(
                self.raw,
                attn_desc.raw,
                grad_mode.into(),
                q_desc.raw,
                queries_ptr,
                k_desc.raw,
                keys_ptr,
                v_desc.raw,
                values_ptr,
                do_desc.raw,
                d_out_ptr,
                weights.len(),
                weights_ptr,
                d_weights_ptr,
                work_space.len(),
                work_space_ptr,
                reserve_space.len(),
                reserve_space_ptr,
            )
            .into_result()
        }
    }
}
