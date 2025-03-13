use crate::{sys, CudnnError, IntoResult};
use std::mem::MaybeUninit;

/// cuDNN library context. It's the central structure required to interact with cuDNN.
/// It holds and manages internal memory allocations.
///
/// # Multi-thread Usage
///
/// While it is technically allowed to use the same context across threads, it is very suboptimal
/// and dangerous so we chose not to expose this functionality. Instead, you should create a context
/// for every thread as also recommended by the
/// [cuDNN docs](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnCreate).
///
/// # Multi-Device Usage
///
/// cuDNN contexts are tied to the current device through the current CUDA context, therefore,
/// for multi-device usage one cuDNN context must be created for every different device.
///
/// # Drop Overhead
///
/// cuDNN contexts hold the internal memory allocations required by the library, and will free those
/// resources on drop. They will also synchronize the entire device when dropping the context.
/// Therefore, you should minimize both the amount of contexts, and the amount of context drops.
/// You should generally create and drop context outside of performance critical code paths.
#[derive(PartialEq, Eq, Hash, Debug)]
pub struct CudnnContext {
    pub(crate) raw: sys::cudnnHandle_t,
}

impl CudnnContext {
    /// Creates a new cuDNN context, allocating the required memory on both host and device.
    ///
    /// cuDNN [docs](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnCreate)
    /// may offer additional information about the APi behavior.
    ///
    /// # Examples
    ///
    /// ```
    /// # use std::error::Error;
    /// #
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// use cudnn::CudnnContext;
    ///
    /// let ctx = CudnnContext::new()?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new() -> Result<Self, CudnnError> {
        let mut raw = MaybeUninit::uninit();

        unsafe {
            sys::cudnnCreate(raw.as_mut_ptr()).into_result()?;
            let raw = raw.assume_init();

            Ok(Self { raw })
        }
    }

    /// Returns the version number of the underlying cuDNN library.
    ///
    /// cuDNN [docs](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnGetVersion)
    /// may offer additional information about the APi behavior.
    pub fn version(&self) -> (u32, u32, u32) {
        unsafe {
            // cudnnGetVersion does not return a state as it never fails.
            let version = sys::cudnnGetVersion();
            (
                (version / 1000) as u32,
                ((version % 1000) / 100) as u32,
                (version % 100) as u32,
            )
        }
    }

    /// Since The same version of a given cuDNN library can be compiled against different CUDA
    /// toolkit versions, this routine returns the CUDA toolkit version that the currently used
    /// cuDNN library has been compiled against.
    ///
    /// cuDNN [docs](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnGetCudartVersion)
    /// may offer additional information about the APi behavior.
    pub fn cuda_version(&self) -> (u32, u32, u32) {
        unsafe {
            // cudnnGetCudartVersion does not return a state as it never fails.
            let version = sys::cudnnGetCudartVersion();
            (
                (version / 1000) as u32,
                ((version % 1000) / 100) as u32,
                (version % 100) as u32,
            )
        }
    }

    /// This function sets the user's CUDA stream in the cuDNN handle.
    ///
    /// The new stream will be used to launch cuDNN GPU kernels or to synchronize to this stream
    /// when cuDNN kernels are launched in the internal streams.
    ///
    /// If the cuDNN library stream is not set, all kernels use the default (NULL) stream.
    /// Setting the user stream in the cuDNN handle guarantees the issue-order execution of cuDNN
    /// calls and other GPU kernels launched in the same stream.
    ///
    /// # Arguments
    ///
    /// `stream` - the CUDA stream to be written to the cuDNN handle.
    ///
    /// cuDNN [docs](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnSetStream)
    /// may offer additional information about the APi behavior.
    ///
    /// # Errors
    ///
    /// Returns error if the supplied stream in invalid or a mismatch if found between the user
    /// stream and the cuDNN handle context.
    pub fn set_stream(&mut self, stream: &cust::stream::Stream) -> Result<(), CudnnError> {
        unsafe {
            sys::cudnnSetStream(self.raw, stream.as_inner() as sys::cudaStream_t).into_result()
        }
    }
}

impl Drop for CudnnContext {
    fn drop(&mut self) {
        unsafe {
            // This can be either a valid cuDNN handle or a null pointer.
            // Since it's getting dropped we shouldn't bother much.
            sys::cudnnDestroy(self.raw);
        }
    }
}
