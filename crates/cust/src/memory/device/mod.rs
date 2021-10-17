use crate::error::CudaResult;
use crate::stream::Stream;

mod device_box;
mod device_buffer;
mod device_slice;

pub use self::device_box::*;
pub use self::device_buffer::*;
pub use self::device_slice::*;

/// Sealed trait implemented by types which can be the source or destination when copying data
/// to/from the device or from one device allocation to another.
pub trait CopyDestination<O: ?Sized>: crate::private::Sealed {
    /// Copy data from `source`. `source` must be the same size as `self`.
    ///
    /// # Errors
    ///
    /// If a CUDA error occurs, return the error.
    fn copy_from(&mut self, source: &O) -> CudaResult<()>;

    /// Copy data to `dest`. `dest` must be the same size as `self`.
    ///
    /// # Errors
    ///
    /// If a CUDA error occurs, return the error.
    fn copy_to(&self, dest: &mut O) -> CudaResult<()>;
}

/// Sealed trait implemented by types which can be the source or destination when copying data
/// asynchronously to/from the device or from one device allocation to another.
///
/// # Safety
///
/// The functions of this trait are unsafe because they return control to the calling code while
/// the copy operation could still be occurring in the background. This could allow calling code
/// to read, modify or deallocate the destination buffer, or to modify or deallocate the source
/// buffer resulting in a data race and undefined behavior.
///
/// Thus to enforce safety, the following invariants must be upheld:
/// * The source and destination are not deallocated
/// * The source is not modified
/// * The destination is not written or read by any other operation
///
/// These invariants must be preserved until the stream is synchronized or an event queued after
/// the copy is triggered.
///
pub trait AsyncCopyDestination<O: ?Sized>: crate::private::Sealed {
    /// Asynchronously copy data from `source`. `source` must be the same size as `self`.
    ///
    /// Host memory used as a source or destination must be page-locked.
    ///
    /// # Safety
    ///
    /// For why this function is unsafe, see [AsyncCopyDestination](trait.AsyncCopyDestination.html)
    ///
    /// # Errors
    ///
    /// If a CUDA error occurs, return the error.
    unsafe fn async_copy_from(&mut self, source: &O, stream: &Stream) -> CudaResult<()>;

    /// Asynchronously copy data to `dest`. `dest` must be the same size as `self`.
    ///
    /// Host memory used as a source or destination must be page-locked.
    ///
    /// # Safety
    ///
    /// For why this function is unsafe, see [AsyncCopyDestination](trait.AsyncCopyDestination.html)
    ///
    /// # Errors
    ///
    /// If a CUDA error occurs, return the error.
    unsafe fn async_copy_to(&self, dest: &mut O, stream: &Stream) -> CudaResult<()>;
}
