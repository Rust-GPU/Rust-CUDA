mod attention_descriptor;
mod seq_data_axis;
mod seq_data_descriptor;

pub use attention_descriptor::*;
pub use seq_data_axis::*;
pub use seq_data_descriptor::*;

use crate::{sys, CudnnContext, CudnnError, DataType, IntoResult};
use cust::memory::GpuBuffer;
use std::mem::MaybeUninit;

impl CudnnContext {
    /// This function computes weight, work, and reserve space buffer sizes used by the following
    /// functions:
    ///
    /// * `multi_head_attn_forward()`
    ///
    /// * `multi_head_attn_backward_data()`
    ///
    /// * `multi_head_attn_backward_weights()`
    ///
    /// # Arguments
    ///
    /// `desc` - multi-head attention descriptor.
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
    /// When `reserve_space` is `None` the function operates in the inference mode in which backward
    /// functions are not invoked, otherwise, the training mode is assumed.
    ///
    /// # Arguments
    ///
    /// * `attn_desc` - multi-head attention descriptor.
    ///
    /// * `current_idx` - time-step in queries to process. When the such argument is negative,
    /// all Q time-steps are processed. When `current_idx` is zero or positive, the forward response
    /// is computed for the selected time-step only.
    ///
    /// * `lo_win_idx` - integer array specifying the start indices of the attention window for
    /// each Q time-step. The start index in K, V sets is inclusive.
    ///
    /// * `hi_win_idx` - integer array specifying the end indices of the attention window for each
    /// Q time-step. The end index is exclusive.
    ///
    /// * `device_seq_lengths_qo` - device array specifying sequence lengths of query, residual,
    /// and output sequence data.
    ///
    /// * `device_seq_lengths_kv` - device array specifying sequence lengths of key and value \
    /// input data.
    ///
    /// * `q_desc` - descriptor for the query and residual sequence data.
    ///
    /// * `queries` - queries data in the device memory.
    ///
    /// * `residuals` - residual data in device memory. Set this argument to `None` if no residual
    /// connections are required.
    ///
    /// * `k_desc` - descriptor for the keys sequence data.
    ///
    /// * `keys` - keys data in device memory.
    ///
    /// * `v_desc` - descriptor for the values sequence data.
    ///
    /// * `values` - values data in device memory.
    ///
    /// * `o_desc` - descriptor for the out sequence data.
    ///
    /// * `out` - out data in device memory.
    ///
    /// * `weights` - weight buffer in device memory.
    ///
    /// * `work_space` - work space buffer in device memory.
    ///
    /// * `reserve_space` - reserve space buffer in device memory. This argument should be `None` in
    /// inference mode.
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
        weights: &impl GpuBuffer<T>,
        work_space: &mut impl GpuBuffer<T>,
        reserve_space: Option<&mut impl GpuBuffer<T>>,
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
}
