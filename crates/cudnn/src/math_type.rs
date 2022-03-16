use crate::sys;

/// Enum stating whether the use of tensor core operations is permitted in a given library routine.
///
/// cuDNN [docs](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnMathType_t)
/// may offer additional information about the APi behavior.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MathType {
    /// Tensor Core operations are not used on pre-NVIDIA A100 GPU devices. On A100 GPU
    /// architecture devices, Tensor Core TF32 operation is permitted.
    Default,
    /// The use of Tensor Core operations is permitted but will not actively perform datatype
    /// down conversion on tensors in order to utilize Tensor Cores.
    TensorOp,
    /// The use of Tensor Core operations is permitted and will actively perform datatype down
    /// conversion on tensors in order to utilize Tensor Cores.
    TensorOpAllowConversion,
    /// Restricted to only kernels that use FMA instructions.
    Fma,
}

impl From<sys::cudnnMathType_t> for MathType {
    fn from(raw: sys::cudnnMathType_t) -> Self {
        match raw {
            sys::cudnnMathType_t::CUDNN_DEFAULT_MATH => Self::Default,
            sys::cudnnMathType_t::CUDNN_TENSOR_OP_MATH => Self::TensorOp,
            sys::cudnnMathType_t::CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION => {
                Self::TensorOpAllowConversion
            }
            sys::cudnnMathType_t::CUDNN_FMA_MATH => Self::Fma,
        }
    }
}

impl From<MathType> for sys::cudnnMathType_t {
    fn from(math_type: MathType) -> Self {
        match math_type {
            MathType::Default => Self::CUDNN_DEFAULT_MATH,
            MathType::TensorOp => Self::CUDNN_TENSOR_OP_MATH,
            MathType::TensorOpAllowConversion => Self::CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION,
            MathType::Fma => Self::CUDNN_FMA_MATH,
        }
    }
}
