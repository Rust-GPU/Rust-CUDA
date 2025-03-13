use crate::{sys, CudnnError, DataType, DropoutDescriptor, IntoResult, MathType, SeqDataType};
use cust::memory::GpuBuffer;
use std::{marker::PhantomData, mem::MaybeUninit};

bitflags::bitflags! {
    /// Miscellaneous switches for configuring auxiliary multi-head attention features.
    pub struct AttnModeFlags: u32 {
        /// Forward declaration of mapping between Q, K and V vectors when the beam size is greater
        /// than one in the Q input. Multiple Q vectors from the same beam bundle map to the **same**
        /// K, V vectors. This means that the beam size in the K, V sets are equal to 1.
        const CUDNN_ATTN_QUERYMAP_ALL_TO_ONE = 0;
        /// Forward declaration of mapping between Q, K and V vectors when the beam size is greater
        /// than one in the Q input. Multiple Q vectors from the same beam bundle map to the **different**
        /// K, V vectors. This requires beam sized in K, V sets to be the same as the Q input.
        const CUDNN_ATTN_QUERYMAP_ONE_TO_ONE = 1;
        /// Use no biases in the attention input and output projections.
        const CUDNN_ATTN_DISABLE_PROJ_BIASES = 0;
        /// Use extra biases in the attention input and output projections.
        const CUDNN_ATTN_ENABLE_PROJ_BIASES = 2;
    }
}

/// A multi-head attention descriptor.
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct AttentionDescriptor<T, U, D1, D2>
where
    T: SeqDataType,
    U: SupportedAttn<T>,
    D1: GpuBuffer<u8>,
    D2: GpuBuffer<u8>,
{
    pub(crate) raw: sys::cudnnAttnDescriptor_t,
    data_type: PhantomData<T>,
    math_prec: PhantomData<U>,
    attn_dropout_desc: DropoutDescriptor<D1>,
    post_dropout_desc: DropoutDescriptor<D2>,
}

impl<T, U, D1, D2> AttentionDescriptor<T, U, D1, D2>
where
    T: SeqDataType,
    U: SupportedAttn<T>,
    D1: GpuBuffer<u8>,
    D2: GpuBuffer<u8>,
{
    /// Creates a new multi-head attention descriptor.
    ///
    /// # Arguments
    ///
    ///   * `mode` -  bit flag enabling various attention options that do not require
    ///     additional numerical values.
    ///   * `n_heads` - number of attention heads.
    ///   * `sm_scaler` - softmax sharpening/smoothing coefficient. Must be positive.
    ///   * `math_type` - nvidia tensor cores setting.
    ///   * `attn_dropout_desc` - descriptor of the dropout operation applied to the
    ///     softmax output.
    ///   * `post_dropout_desc` - descriptor of the dropout operation applied to the
    ///     multi-head attention output, just before the point where residual
    ///     connections are added.
    ///   * `q_size` - q vectors length.
    ///   * `k_size` - k vectors length.
    ///   * `v_size` - v vectors length.
    ///   * `q_proj_size` - q vectors length after input projection.
    ///   * `k_proj_size` - k vectors length after input projection.
    ///   * `v_proj_size` - v vectors length after input projection.
    ///   * `o_proj_size` - h vectors length after output projection.
    ///   * `qo_max_seq_length` - largest sequence length expected in sequence data
    ///     descriptors related to Q, O, dQ and dO inputs and outputs.
    ///   * `kv_max_seq_length` - largest sequence length expected in sequence data
    ///     descriptors related to K, V, dK and dV inputs and outputs.
    ///   * `max_batch_size` - largest batch expected in any sequential data descriptor.
    ///   * `max_bream_size` - largest beam expected in any sequential data descriptor.
    ///
    /// cuDNN
    /// [docs](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnSetAttnDescriptor)
    /// may offer additional information about the APi behavior.
    ///
    /// # Errors
    ///
    /// Returns errors if an unsupported combination of arguments is detected. Some
    /// examples include:
    ///
    ///   * post projection Q and K are not equal.
    ///   * math type is not supported.
    ///   * one or more of the following arguments were either negative or zero:
    ///     `n_heads`, `q_size`, `k_size`, `v_size`, `qo_max_seq_length`,
    ///     `kv_max_seq_length`, `max_batch_size` and ` max_beam_size`.
    ///   * one or more of the following arguments were negative: `q_proj_size`,
    ///     `k_proj_size`, `v_proj_size`, `sm_scaler`.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        mode: AttnModeFlags,
        n_heads: i32,
        sm_scaler: f64,
        math_type: MathType,
        attn_dropout_desc: DropoutDescriptor<D1>,
        post_dropout_desc: DropoutDescriptor<D2>,
        q_size: i32,
        k_size: i32,
        v_size: i32,
        q_proj_size: impl Into<Option<i32>>,
        k_proj_size: impl Into<Option<i32>>,
        v_proj_size: impl Into<Option<i32>>,
        o_proj_size: impl Into<Option<i32>>,
        qo_max_seq_length: i32,
        kv_max_seq_lenght: i32,
        max_batch_size: i32,
        max_beam_size: i32,
    ) -> Result<Self, CudnnError> {
        let mut raw = MaybeUninit::uninit();

        unsafe {
            sys::cudnnCreateAttnDescriptor(raw.as_mut_ptr()).into_result()?;

            let raw = raw.assume_init();

            sys::cudnnSetAttnDescriptor(
                raw,
                mode.bits(),
                n_heads,
                sm_scaler,
                T::into_raw(),
                U::into_raw(),
                math_type.into(),
                attn_dropout_desc.raw,
                post_dropout_desc.raw,
                q_size,
                k_size,
                v_size,
                q_proj_size.into().unwrap_or(0),
                k_proj_size.into().unwrap_or(0),
                v_proj_size.into().unwrap_or(0),
                o_proj_size.into().unwrap_or(0),
                qo_max_seq_length,
                kv_max_seq_lenght,
                max_batch_size,
                max_beam_size,
            )
            .into_result()?;

            Ok(Self {
                raw,
                data_type: PhantomData,
                math_prec: PhantomData,
                attn_dropout_desc,
                post_dropout_desc,
            })
        }
    }
}

impl<T, U, D1, D2> Drop for AttentionDescriptor<T, U, D1, D2>
where
    T: SeqDataType,
    U: SupportedAttn<T>,
    D1: GpuBuffer<u8>,
    D2: GpuBuffer<u8>,
{
    fn drop(&mut self) {
        unsafe {
            sys::cudnnDestroyAttnDescriptor(self.raw);
        }
    }
}

/// Controls the compute math precision in the multi-head attention. The following
/// applies:
///
///   * For input and output in `f32`, the math precision of the layer can only be `f32`.
///   * For input and output in `f64` the math precision of the layer can only be `f64`.
pub trait SupportedAttn<T>
where
    Self: DataType,
    T: SeqDataType,
{
}

impl SupportedAttn<f32> for f32 {}
impl SupportedAttn<f64> for f64 {}
