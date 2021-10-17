use crate::error::{CudaResult, DropResult, ToResult};
use crate::memory::device::{AsyncCopyDestination, CopyDestination, DSlice};
use crate::memory::malloc::{cuda_free, cuda_malloc};
use crate::memory::DeviceCopy;
use crate::memory::DevicePointer;
use crate::stream::Stream;
use crate::sys as cuda;
use std::mem;
use std::ops::{Deref, DerefMut};

use std::ptr;

/// Fixed-size device-side buffer. Provides basic access to device memory.
#[derive(Debug)]
pub struct DBuffer<T> {
    buf: DevicePointer<T>,
    capacity: usize,
}
impl<T> DBuffer<T> {
    /// Allocate a new device buffer large enough to hold `size` `T`'s, but without
    /// initializing the contents.
    ///
    /// # Errors
    ///
    /// If the allocation fails, returns the error from CUDA. If `size` is large enough that
    /// `size * mem::sizeof::<T>()` overflows usize, then returns InvalidMemoryAllocation.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the contents of the buffer are initialized before reading from
    /// the buffer.
    ///
    /// # Examples
    ///
    /// ```
    /// # let _context = cust::quick_init().unwrap();
    /// use cust::memory::*;
    /// let mut buffer = unsafe { DeviceBuffer::uninitialized(5).unwrap() };
    /// buffer.copy_from(&[0u64, 1, 2, 3, 4]).unwrap();
    /// ```
    pub unsafe fn uninitialized(size: usize) -> CudaResult<Self> {
        let ptr = if size > 0 && mem::size_of::<T>() > 0 {
            cuda_malloc(size)?
        } else {
            DevicePointer::wrap(ptr::NonNull::dangling().as_ptr() as *mut T)
        };
        Ok(DBuffer {
            buf: ptr,
            capacity: size,
        })
    }

    /// Allocate a new device buffer large enough to hold `size` `T`'s and fill the contents with
    /// zeroes (`0u8`).
    ///
    /// # Errors
    ///
    /// If the allocation fails, returns the error from CUDA. If `size` is large enough that
    /// `size * mem::sizeof::<T>()` overflows usize, then returns InvalidMemoryAllocation.
    ///
    /// # Safety
    ///
    /// The backing memory is zeroed, which may not be a valid bit-pattern for type `T`. The caller
    /// must ensure either that all-zeroes is a valid bit-pattern for type `T` or that the backing
    /// memory is set to a valid value before it is read.
    ///
    /// # Examples
    ///
    /// ```
    /// # let _context = cust::quick_init().unwrap();
    /// use cust::memory::*;
    /// let buffer = unsafe { DeviceBuffer::zeroed(5).unwrap() };
    /// let mut host_values = [1u64, 2, 3, 4, 5];
    /// buffer.copy_to(&mut host_values).unwrap();
    /// assert_eq!([0u64, 0, 0, 0, 0], host_values);
    /// ```
    pub unsafe fn zeroed(size: usize) -> CudaResult<Self> {
        let ptr = if size > 0 && mem::size_of::<T>() > 0 {
            let mut ptr = cuda_malloc(size)?;
            cuda::cuMemsetD8_v2(ptr.as_raw_mut() as u64, 0, size * mem::size_of::<T>())
                .to_result()?;
            ptr
        } else {
            DevicePointer::wrap(ptr::NonNull::dangling().as_ptr() as *mut T)
        };
        Ok(DBuffer {
            buf: ptr,
            capacity: size,
        })
    }

    /// Creates a `DeviceBuffer<T>` directly from the raw components of another device buffer.
    ///
    /// # Safety
    ///
    /// This is highly unsafe, due to the number of invariants that aren't
    /// checked:
    ///
    /// * `ptr` needs to have been previously allocated via `DeviceBuffer` or
    /// [`cuda_malloc`](fn.cuda_malloc.html).
    /// * `ptr`'s `T` needs to have the same size and alignment as it was allocated with.
    /// * `capacity` needs to be the capacity that the pointer was allocated with.
    ///
    /// Violating these may cause problems like corrupting the CUDA driver's
    /// internal data structures.
    ///
    /// The ownership of `ptr` is effectively transferred to the
    /// `DeviceBuffer<T>` which may then deallocate, reallocate or change the
    /// contents of memory pointed to by the pointer at will. Ensure
    /// that nothing else uses the pointer after calling this
    /// function.
    ///
    /// # Examples
    ///
    /// ```
    /// # let _context = cust::quick_init().unwrap();
    /// use std::mem;
    /// use cust::memory::*;
    ///
    /// let mut buffer = DeviceBuffer::from_slice(&[0u64; 5]).unwrap();
    /// let ptr = buffer.as_device_ptr();
    /// let size = buffer.len();
    ///
    /// mem::forget(buffer);
    ///
    /// let buffer = unsafe { DeviceBuffer::from_raw_parts(ptr, size) };
    /// ```
    pub unsafe fn from_raw_parts(ptr: DevicePointer<T>, capacity: usize) -> DBuffer<T> {
        DBuffer { buf: ptr, capacity }
    }

    /// Destroy a `DeviceBuffer`, returning an error.
    ///
    /// Deallocating device memory can return errors from previous asynchronous work. This function
    /// destroys the given buffer and returns the error and the un-destroyed buffer on failure.
    ///
    /// # Example
    ///
    /// ```
    /// # let _context = cust::quick_init().unwrap();
    /// use cust::memory::*;
    /// let x = DeviceBuffer::from_slice(&[10, 20, 30]).unwrap();
    /// match DeviceBuffer::drop(x) {
    ///     Ok(()) => println!("Successfully destroyed"),
    ///     Err((e, buf)) => {
    ///         println!("Failed to destroy buffer: {:?}", e);
    ///         // Do something with buf
    ///     },
    /// }
    /// ```
    pub fn drop(mut dev_buf: DBuffer<T>) -> DropResult<DBuffer<T>> {
        if dev_buf.buf.is_null() {
            return Ok(());
        }

        if dev_buf.capacity > 0 && mem::size_of::<T>() > 0 {
            let capacity = dev_buf.capacity;
            let ptr = mem::replace(&mut dev_buf.buf, DevicePointer::null());
            unsafe {
                match cuda_free(ptr) {
                    Ok(()) => {
                        mem::forget(dev_buf);
                        Ok(())
                    }
                    Err(e) => Err((e, DBuffer::from_raw_parts(ptr, capacity))),
                }
            }
        } else {
            Ok(())
        }
    }
}
impl<T: DeviceCopy> DBuffer<T> {
    /// Allocate a new device buffer of the same size as `slice`, initialized with a clone of
    /// the data in `slice`.
    ///
    /// # Errors
    ///
    /// If the allocation fails, returns the error from CUDA.
    ///
    /// # Examples
    ///
    /// ```
    /// # let _context = cust::quick_init().unwrap();
    /// use cust::memory::*;
    /// let values = [0u64; 5];
    /// let mut buffer = DeviceBuffer::from_slice(&values).unwrap();
    /// ```
    pub fn from_slice(slice: &[T]) -> CudaResult<Self> {
        unsafe {
            let mut uninit = DBuffer::uninitialized(slice.len())?;
            uninit.copy_from(slice)?;
            Ok(uninit)
        }
    }

    /// Asynchronously allocate a new buffer of the same size as `slice`, initialized
    /// with a clone of the data in `slice`.
    ///
    /// # Safety
    ///
    /// For why this function is unsafe, see [AsyncCopyDestination](trait.AsyncCopyDestination.html)
    ///
    /// # Errors
    ///
    /// If the allocation fails, returns the error from CUDA.
    ///
    /// # Examples
    ///
    /// ```
    /// # let _context = cust::quick_init().unwrap();
    /// use cust::memory::*;
    /// use cust::stream::{Stream, StreamFlags};
    ///
    /// let stream = Stream::new(StreamFlags::NON_BLOCKING, None).unwrap();
    /// let values = [0u64; 5];
    /// unsafe {
    ///     let mut buffer = DeviceBuffer::from_slice_async(&values, &stream).unwrap();
    ///     stream.synchronize();
    ///     // Perform some operation on the buffer
    /// }
    /// ```
    pub unsafe fn from_slice_async(slice: &[T], stream: &Stream) -> CudaResult<Self> {
        let mut uninit = DBuffer::uninitialized(slice.len())?;
        uninit.async_copy_from(slice, stream)?;
        Ok(uninit)
    }
}
impl<T> Deref for DBuffer<T> {
    type Target = DSlice<T>;

    fn deref(&self) -> &DSlice<T> {
        unsafe {
            DSlice::from_slice(::std::slice::from_raw_parts(
                self.buf.as_raw(),
                self.capacity,
            ))
        }
    }
}
impl<T> DerefMut for DBuffer<T> {
    fn deref_mut(&mut self) -> &mut DSlice<T> {
        unsafe {
            &mut *(::std::slice::from_raw_parts_mut(self.buf.as_raw_mut(), self.capacity)
                as *mut [T] as *mut DSlice<T>)
        }
    }
}
impl<T> Drop for DBuffer<T> {
    fn drop(&mut self) {
        if self.buf.is_null() {
            return;
        }

        if self.capacity > 0 && mem::size_of::<T>() > 0 {
            let ptr = mem::replace(&mut self.buf, DevicePointer::null());
            unsafe {
                let _ = cuda_free(ptr);
            }
        }
        self.capacity = 0;
    }
}

#[cfg(test)]
mod test_device_buffer {
    use super::*;
    use crate::memory::device::DBox;
    use crate::stream::{Stream, StreamFlags};

    #[derive(Clone, Copy, Debug)]
    struct ZeroSizedType;
    unsafe impl DeviceCopy for ZeroSizedType {}

    #[test]
    fn test_from_slice_drop() {
        let _context = crate::quick_init().unwrap();
        let buf = DBuffer::from_slice(&[0u64, 1, 2, 3, 4, 5]).unwrap();
        drop(buf);
    }

    #[test]
    fn test_copy_to_from_device() {
        let _context = crate::quick_init().unwrap();
        let start = [0u64, 1, 2, 3, 4, 5];
        let mut end = [0u64, 0, 0, 0, 0, 0];
        let buf = DBuffer::from_slice(&start).unwrap();
        buf.copy_to(&mut end).unwrap();
        assert_eq!(start, end);
    }

    #[test]
    fn test_async_copy_to_from_device() {
        let _context = crate::quick_init().unwrap();
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None).unwrap();
        let start = [0u64, 1, 2, 3, 4, 5];
        let mut end = [0u64, 0, 0, 0, 0, 0];
        unsafe {
            let buf = DBuffer::from_slice_async(&start, &stream).unwrap();
            buf.async_copy_to(&mut end, &stream).unwrap();
        }
        stream.synchronize().unwrap();
        assert_eq!(start, end);
    }

    #[test]
    fn test_slice() {
        let _context = crate::quick_init().unwrap();
        let start = [0u64, 1, 2, 3, 4, 5];
        let mut end = [0u64, 0];
        let mut buf = DBuffer::from_slice(&[0u64, 0, 0, 0]).unwrap();
        buf.copy_from(&start[0..4]).unwrap();
        buf[0..2].copy_to(&mut end).unwrap();
        assert_eq!(start[0..2], end);
    }

    #[test]
    fn test_async_slice() {
        let _context = crate::quick_init().unwrap();
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None).unwrap();
        let start = [0u64, 1, 2, 3, 4, 5];
        let mut end = [0u64, 0];
        unsafe {
            let mut buf = DBuffer::from_slice_async(&[0u64, 0, 0, 0], &stream).unwrap();
            buf.async_copy_from(&start[0..4], &stream).unwrap();
            buf[0..2].async_copy_to(&mut end, &stream).unwrap();
            stream.synchronize().unwrap();
            assert_eq!(start[0..2], end);
        }
    }

    #[test]
    #[should_panic]
    fn test_copy_to_d2h_wrong_size() {
        let _context = crate::quick_init().unwrap();
        let buf = DBuffer::from_slice(&[0u64, 1, 2, 3, 4, 5]).unwrap();
        let mut end = [0u64, 1, 2, 3, 4];
        let _ = buf.copy_to(&mut end);
    }

    #[test]
    #[should_panic]
    fn test_async_copy_to_d2h_wrong_size() {
        let _context = crate::quick_init().unwrap();
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None).unwrap();
        unsafe {
            let buf = DBuffer::from_slice_async(&[0u64, 1, 2, 3, 4, 5], &stream).unwrap();
            let mut end = [0u64, 1, 2, 3, 4];
            let _ = buf.async_copy_to(&mut end, &stream);
        }
    }

    #[test]
    #[should_panic]
    fn test_copy_from_h2d_wrong_size() {
        let _context = crate::quick_init().unwrap();
        let start = [0u64, 1, 2, 3, 4];
        let mut buf = DBuffer::from_slice(&[0u64, 1, 2, 3, 4, 5]).unwrap();
        let _ = buf.copy_from(&start);
    }

    #[test]
    #[should_panic]
    fn test_async_copy_from_h2d_wrong_size() {
        let _context = crate::quick_init().unwrap();
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None).unwrap();
        let start = [0u64, 1, 2, 3, 4];
        unsafe {
            let mut buf = DBuffer::from_slice_async(&[0u64, 1, 2, 3, 4, 5], &stream).unwrap();
            let _ = buf.async_copy_from(&start, &stream);
        }
    }

    #[test]
    fn test_copy_device_slice_to_device() {
        let _context = crate::quick_init().unwrap();
        let start = DBuffer::from_slice(&[0u64, 1, 2, 3, 4, 5]).unwrap();
        let mut mid = DBuffer::from_slice(&[0u64, 0, 0, 0]).unwrap();
        let mut end = DBuffer::from_slice(&[0u64, 0]).unwrap();
        let mut host_end = [0u64, 0];
        start[1..5].copy_to(&mut mid).unwrap();
        end.copy_from(&mid[1..3]).unwrap();
        end.copy_to(&mut host_end).unwrap();
        assert_eq!([2u64, 3], host_end);
    }

    #[test]
    fn test_async_copy_device_slice_to_device() {
        let _context = crate::quick_init().unwrap();
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None).unwrap();
        unsafe {
            let start = DBuffer::from_slice_async(&[0u64, 1, 2, 3, 4, 5], &stream).unwrap();
            let mut mid = DBuffer::from_slice_async(&[0u64, 0, 0, 0], &stream).unwrap();
            let mut end = DBuffer::from_slice_async(&[0u64, 0], &stream).unwrap();
            let mut host_end = [0u64, 0];
            start[1..5].async_copy_to(&mut mid, &stream).unwrap();
            end.async_copy_from(&mid[1..3], &stream).unwrap();
            end.async_copy_to(&mut host_end, &stream).unwrap();
            stream.synchronize().unwrap();
            assert_eq!([2u64, 3], host_end);
        }
    }

    #[test]
    #[should_panic]
    fn test_copy_to_d2d_wrong_size() {
        let _context = crate::quick_init().unwrap();
        let buf = DBuffer::from_slice(&[0u64, 1, 2, 3, 4, 5]).unwrap();
        let mut end = DBuffer::from_slice(&[0u64, 1, 2, 3, 4]).unwrap();
        let _ = buf.copy_to(&mut end);
    }

    #[test]
    #[should_panic]
    fn test_async_copy_to_d2d_wrong_size() {
        let _context = crate::quick_init().unwrap();
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None).unwrap();
        unsafe {
            let buf = DBuffer::from_slice_async(&[0u64, 1, 2, 3, 4, 5], &stream).unwrap();
            let mut end = DBuffer::from_slice_async(&[0u64, 1, 2, 3, 4], &stream).unwrap();
            let _ = buf.async_copy_to(&mut end, &stream);
        }
    }

    #[test]
    #[should_panic]
    fn test_copy_from_d2d_wrong_size() {
        let _context = crate::quick_init().unwrap();
        let mut buf = DBuffer::from_slice(&[0u64, 1, 2, 3, 4, 5]).unwrap();
        let start = DBuffer::from_slice(&[0u64, 1, 2, 3, 4]).unwrap();
        let _ = buf.copy_from(&start);
    }

    #[test]
    #[should_panic]
    fn test_async_copy_from_d2d_wrong_size() {
        let _context = crate::quick_init().unwrap();
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None).unwrap();
        unsafe {
            let mut buf = DBuffer::from_slice_async(&[0u64, 1, 2, 3, 4, 5], &stream).unwrap();
            let start = DBuffer::from_slice_async(&[0u64, 1, 2, 3, 4], &stream).unwrap();
            let _ = buf.async_copy_from(&start, &stream);
        }
    }

    #[test]
    fn test_can_create_uninitialized_non_devicecopy_buffers() {
        let _context = crate::quick_init().unwrap();
        unsafe {
            let _box: DBox<Vec<u8>> = DBox::uninitialized().unwrap();
            let buffer: DBuffer<Vec<u8>> = DBuffer::uninitialized(10).unwrap();
            let _slice = &buffer[0..5];
        }
    }
}
