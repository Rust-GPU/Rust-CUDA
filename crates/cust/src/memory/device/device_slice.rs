use crate::error::{CudaResult, ToResult};
use crate::memory::device::AsyncCopyDestination;
use crate::memory::device::{CopyDestination, DeviceBuffer};
use crate::memory::DeviceCopy;
use crate::memory::DevicePointer;
use crate::stream::Stream;
use crate::sys as cuda;
use std::mem;

use std::os::raw::c_void;

/// Fixed-size device-side slice.
#[derive(Debug)]
#[repr(C)]
pub struct DeviceSlice<T: DeviceCopy> {
    ptr: DevicePointer<T>,
    len: usize,
}

impl<T: DeviceCopy + Default + Clone> DeviceSlice<T> {
    pub fn as_host_vec(&self) -> CudaResult<Vec<T>> {
        let mut vec = vec![T::default(); self.len()];
        self.copy_to(&mut vec)?;
        Ok(vec)
    }
}

// This works by faking a regular slice out of the device raw-pointer and the length and transmuting
// I have no idea if this is safe or not. Probably not, though I can't imagine how the compiler
// could possibly know that the pointer is not de-referenceable. I'm banking that we get proper
// Dynamicaly-sized Types before the compiler authors break this assumption.
impl<T: DeviceCopy> DeviceSlice<T> {
    /// Returns the number of elements in the slice.
    ///
    /// # Examples
    ///
    /// ```
    /// # let _context = cust::quick_init().unwrap();
    /// use cust::memory::*;
    /// let a = DeviceBuffer::from_slice(&[1, 2, 3]).unwrap();
    /// assert_eq!(a.len(), 3);
    /// ```
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the slice has a length of 0.
    ///
    /// # Examples
    ///
    /// ```
    /// # let _context = cust::quick_init().unwrap();
    /// use cust::memory::*;
    /// let a : DeviceBuffer<u64> = unsafe { DeviceBuffer::uninitialized(0).unwrap() };
    /// assert!(a.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Return a raw device-pointer to the slice's buffer.
    ///
    /// The caller must ensure that the slice outlives the pointer this function returns, or else
    /// it will end up pointing to garbage. The caller must also ensure that the pointer is not
    /// dereferenced by the CPU.
    ///
    /// Examples:
    ///
    /// ```
    /// # let _context = cust::quick_init().unwrap();
    /// use cust::memory::*;
    /// let a = DeviceBuffer::from_slice(&[1, 2, 3]).unwrap();
    /// println!("{:p}", a.as_ptr());
    /// ```
    pub fn as_device_ptr(&self) -> DevicePointer<T> {
        self.ptr
    }

    /* TODO (AL): keep these?
    /// Divides one DeviceSlice into two at a given index.
    ///
    /// The first will contain all indices from `[0, mid)` (excluding the index `mid` itself) and
    /// the second will contain all indices from `[mid, len)` (excluding the index `len` itself).
    ///
    /// # Panics
    ///
    /// Panics if `min > len`.
    ///
    /// Examples:
    ///
    /// ```
    /// # let _context = cust::quick_init().unwrap();
    /// use cust::memory::*;
    /// let buf = DeviceBuffer::from_slice(&[0u64, 1, 2, 3, 4, 5]).unwrap();
    /// let (left, right) = buf.split_at(3);
    /// let mut left_host = [0u64, 0, 0];
    /// let mut right_host = [0u64, 0, 0];
    /// left.copy_to(&mut left_host).unwrap();
    /// right.copy_to(&mut right_host).unwrap();
    /// assert_eq!([0u64, 1, 2], left_host);
    /// assert_eq!([3u64, 4, 5], right_host);
    /// ```
    pub fn split_at(&self, mid: usize) -> (&DeviceSlice<T>, &DeviceSlice<T>) {
        let (left, right) = self.0.split_at(mid);
        unsafe {
            (
                DeviceSlice::from_slice(left),
                DeviceSlice::from_slice(right),
            )
        }
    }

    /// Divides one mutable DeviceSlice into two at a given index.
    ///
    /// The first will contain all indices from `[0, mid)` (excluding the index `mid` itself) and
    /// the second will contain all indices from `[mid, len)` (excluding the index `len` itself).
    ///
    /// # Panics
    ///
    /// Panics if `min > len`.
    ///
    /// Examples:
    ///
    /// ```
    /// # let _context = cust::quick_init().unwrap();
    /// use cust::memory::*;
    /// let mut buf = DeviceBuffer::from_slice(&[0u64, 0, 0, 0, 0, 0]).unwrap();
    ///
    /// {
    ///     let (left, right) = buf.split_at_mut(3);
    ///     let left_host = [0u64, 1, 2];
    ///     let right_host = [3u64, 4, 5];
    ///     left.copy_from(&left_host).unwrap();
    ///     right.copy_from(&right_host).unwrap();
    /// }
    ///
    /// let mut host_full = [0u64; 6];
    /// buf.copy_to(&mut host_full).unwrap();
    /// assert_eq!([0u64, 1, 2, 3, 4, 5], host_full);
    /// ```
    pub fn split_at_mut(&mut self, mid: usize) -> (&mut DeviceSlice<T>, &mut DeviceSlice<T>) {
        let (left, right) = self.0.split_at_mut(mid);
        unsafe {
            (
                DeviceSlice::from_slice_mut(left),
                DeviceSlice::from_slice_mut(right),
            )
        }
    }
    */

    /// Forms a slice from a `DevicePointer` and a length.
    ///
    /// The `len` argument is the number of _elements_, not the number of bytes.
    ///
    /// # Safety
    ///
    /// This function is unsafe as there is no guarantee that the given pointer is valid for `len`
    /// elements, nor whether the lifetime inferred is a suitable lifetime for the returned slice.
    ///
    /// # Caveat
    ///
    /// The lifetime for the returned slice is inferred from its usage. To prevent accidental misuse,
    /// it's suggested to tie the lifetime to whatever source lifetime is safe in the context, such
    /// as by providing a helper function taking the lifetime of a host value for the slice or
    /// by explicit annotation.
    ///
    /// # Examples
    ///
    /// ```
    /// # let _context = cust::quick_init().unwrap();
    /// use cust::memory::*;
    /// let mut x = DeviceBuffer::from_slice(&[0u64, 1, 2, 3, 4, 5]).unwrap();
    /// // Manually slice the buffer (this is not recommended!)
    /// let ptr = unsafe { x.as_device_ptr().offset(1) };
    /// let slice = unsafe { DeviceSlice::from_raw_parts(ptr, 2) };
    /// let mut host_buf = [0u64, 0];
    /// slice.copy_to(&mut host_buf).unwrap();
    /// assert_eq!([1u64, 2], host_buf);
    /// ```
    #[allow(clippy::needless_pass_by_value)]
    pub unsafe fn from_raw_parts(ptr: DevicePointer<T>, len: usize) -> DeviceSlice<T> {
        DeviceSlice { ptr, len }
    }

    /// Performs the same functionality as `from_raw_parts`, except that a
    /// mutable slice is returned.
    ///
    /// # Safety
    ///
    /// This function is unsafe as there is no guarantee that the given pointer is valid for `len`
    /// elements, nor whether the lifetime inferred is a suitable lifetime for the returned slice.
    ///
    /// This function is unsafe as there is no guarantee that the given pointer is valid for `len`
    /// elements, not whether the lifetime inferred is a suitable lifetime for the returned slice,
    /// as well as not being able to provide a non-aliasing guarantee of the returned
    /// mutable slice. `data` must be non-null and aligned even for zero-length
    /// slices as with `from_raw_parts`.
    ///
    /// See the documentation of `from_raw_parts` for more details.
    pub unsafe fn from_raw_parts_mut(ptr: DevicePointer<T>, len: usize) -> DeviceSlice<T> {
        DeviceSlice { ptr, len }
    }
}

impl<T: DeviceCopy> crate::private::Sealed for DeviceSlice<T> {}
impl<T: DeviceCopy, I: AsRef<[T]> + AsMut<[T]> + ?Sized> CopyDestination<I> for DeviceSlice<T> {
    fn copy_from(&mut self, val: &I) -> CudaResult<()> {
        let val = val.as_ref();
        assert!(
            self.len() == val.len(),
            "destination and source slices have different lengths"
        );
        let size = mem::size_of::<T>() * self.len();
        if size != 0 {
            unsafe {
                cuda::cuMemcpyHtoD_v2(self.ptr.as_raw(), val.as_ptr() as *const c_void, size)
                    .to_result()?
            }
        }
        Ok(())
    }

    fn copy_to(&self, val: &mut I) -> CudaResult<()> {
        let val = val.as_mut();
        assert!(
            self.len() == val.len(),
            "destination and source slices have different lengths"
        );
        let size = mem::size_of::<T>() * self.len();
        if size != 0 {
            unsafe {
                cuda::cuMemcpyDtoH_v2(
                    val.as_mut_ptr() as *mut c_void,
                    self.as_device_ptr().as_raw(),
                    size,
                )
                .to_result()?
            }
        }
        Ok(())
    }
}
impl<T: DeviceCopy> CopyDestination<DeviceSlice<T>> for DeviceSlice<T> {
    fn copy_from(&mut self, val: &DeviceSlice<T>) -> CudaResult<()> {
        assert!(
            self.len() == val.len(),
            "destination and source slices have different lengths"
        );
        let size = mem::size_of::<T>() * self.len();
        if size != 0 {
            unsafe {
                cuda::cuMemcpyDtoD_v2(self.ptr.as_raw(), val.as_device_ptr().as_raw(), size)
                    .to_result()?
            }
        }
        Ok(())
    }

    fn copy_to(&self, val: &mut DeviceSlice<T>) -> CudaResult<()> {
        assert!(
            self.len() == val.len(),
            "destination and source slices have different lengths"
        );
        let size = mem::size_of::<T>() * self.len();
        if size != 0 {
            unsafe {
                cuda::cuMemcpyDtoD_v2(
                    val.as_device_ptr().as_raw(),
                    self.as_device_ptr().as_raw(),
                    size,
                )
                .to_result()?
            }
        }
        Ok(())
    }
}
impl<T: DeviceCopy> CopyDestination<DeviceBuffer<T>> for DeviceSlice<T> {
    fn copy_from(&mut self, val: &DeviceBuffer<T>) -> CudaResult<()> {
        self.copy_from(val as &DeviceSlice<T>)
    }

    fn copy_to(&self, val: &mut DeviceBuffer<T>) -> CudaResult<()> {
        self.copy_to(val as &mut DeviceSlice<T>)
    }
}
impl<T: DeviceCopy, I: AsRef<[T]> + AsMut<[T]> + ?Sized> AsyncCopyDestination<I>
    for DeviceSlice<T>
{
    unsafe fn async_copy_from(&mut self, val: &I, stream: &Stream) -> CudaResult<()> {
        let val = val.as_ref();
        assert!(
            self.len() == val.len(),
            "destination and source slices have different lengths"
        );
        let size = mem::size_of::<T>() * self.len();
        if size != 0 {
            cuda::cuMemcpyHtoDAsync_v2(
                self.ptr.as_raw(),
                val.as_ptr() as *const c_void,
                size,
                stream.as_inner(),
            )
            .to_result()?
        }
        Ok(())
    }

    unsafe fn async_copy_to(&self, val: &mut I, stream: &Stream) -> CudaResult<()> {
        let val = val.as_mut();
        assert!(
            self.len() == val.len(),
            "destination and source slices have different lengths"
        );
        let size = mem::size_of::<T>() * self.len();
        if size != 0 {
            cuda::cuMemcpyDtoHAsync_v2(
                val.as_mut_ptr() as *mut c_void,
                self.as_device_ptr().as_raw(),
                size,
                stream.as_inner(),
            )
            .to_result()?
        }
        Ok(())
    }
}
impl<T: DeviceCopy> AsyncCopyDestination<DeviceSlice<T>> for DeviceSlice<T> {
    unsafe fn async_copy_from(&mut self, val: &DeviceSlice<T>, stream: &Stream) -> CudaResult<()> {
        assert!(
            self.len() == val.len(),
            "destination and source slices have different lengths"
        );
        let size = mem::size_of::<T>() * self.len();
        if size != 0 {
            cuda::cuMemcpyDtoDAsync_v2(
                self.as_device_ptr().as_raw(),
                val.as_device_ptr().as_raw(),
                size,
                stream.as_inner(),
            )
            .to_result()?
        }
        Ok(())
    }

    unsafe fn async_copy_to(&self, val: &mut DeviceSlice<T>, stream: &Stream) -> CudaResult<()> {
        assert!(
            self.len() == val.len(),
            "destination and source slices have different lengths"
        );
        let size = mem::size_of::<T>() * self.len();
        if size != 0 {
            cuda::cuMemcpyDtoDAsync_v2(
                val.as_device_ptr().as_raw(),
                self.as_device_ptr().as_raw(),
                size,
                stream.as_inner(),
            )
            .to_result()?
        }
        Ok(())
    }
}
impl<T: DeviceCopy> AsyncCopyDestination<DeviceBuffer<T>> for DeviceSlice<T> {
    unsafe fn async_copy_from(&mut self, val: &DeviceBuffer<T>, stream: &Stream) -> CudaResult<()> {
        self.async_copy_from(val as &DeviceSlice<T>, stream)
    }

    unsafe fn async_copy_to(&self, val: &mut DeviceBuffer<T>, stream: &Stream) -> CudaResult<()> {
        self.async_copy_to(val as &mut DeviceSlice<T>, stream)
    }
}
