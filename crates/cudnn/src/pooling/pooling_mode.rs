use crate::sys;

/// Specifies the pooling method.
///
/// cuDNN [docs](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnPoolingMode_t)
/// may offer additional information about the APi behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PoolingMode {
    /// The maximum value inside the pooling window is used.
    Max,
    /// Values inside the pooling window are averaged. The number of elements used to calculate
    /// the average includes spatial locations falling in the padding region.
    AvgIncludePadding,
    /// Values inside the pooling window are averaged. The number of elements used to calculate
    /// the average excludes spatial locations falling in the padding region.
    AvgExcludePadding,
    /// The maximum value inside the pooling window is used. The algorithm used is deterministic.
    MaxDeterministic,
}

impl From<PoolingMode> for sys::cudnnPoolingMode_t {
    fn from(mode: PoolingMode) -> Self {
        match mode {
            PoolingMode::Max => Self::CUDNN_POOLING_MAX,
            PoolingMode::AvgExcludePadding => Self::CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING,
            PoolingMode::AvgIncludePadding => Self::CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING,
            PoolingMode::MaxDeterministic => Self::CUDNN_POOLING_MAX_DETERMINISTIC,
        }
    }
}
