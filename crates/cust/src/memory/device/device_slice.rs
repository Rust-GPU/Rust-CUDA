use crate::error::{CudaResult, ToResult};
use crate::memory::device::AsyncCopyDestination;
use crate::memory::device::{CopyDestination, DBuffer};
use crate::memory::DeviceCopy;
use crate::memory::DevicePointer;
use crate::stream::Stream;
use crate::sys as cuda;
use std::iter::{ExactSizeIterator, FusedIterator};
use std::mem::{self};
use std::ops::{
    Index, IndexMut, Range, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive,
};

use std::os::raw::c_void;
use std::slice::{self, Chunks, ChunksMut};

/// Fixed-size device-side slice.
#[derive(Debug)]
#[repr(C)]
pub struct DSlice<T>([T]);

impl<T: DeviceCopy + Default + Clone> DSlice<T> {
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
impl<T> DSlice<T> {
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
        self.0.len()
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
        self.0.is_empty()
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
    pub fn as_ptr(&self) -> *const T {
        self.0.as_ptr()
    }

    /// Returns an unsafe mutable device-pointer to the slice's buffer.
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
    /// let mut a = DeviceBuffer::from_slice(&[1, 2, 3]).unwrap();
    /// println!("{:p}", a.as_mut_ptr());
    /// ```
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.0.as_mut_ptr()
    }

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
    pub fn split_at(&self, mid: usize) -> (&DSlice<T>, &DSlice<T>) {
        let (left, right) = self.0.split_at(mid);
        unsafe { (DSlice::from_slice(left), DSlice::from_slice(right)) }
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
    pub fn split_at_mut(&mut self, mid: usize) -> (&mut DSlice<T>, &mut DSlice<T>) {
        let (left, right) = self.0.split_at_mut(mid);
        unsafe { (DSlice::from_slice_mut(left), DSlice::from_slice_mut(right)) }
    }

    /// Returns an iterator over `chunk_size` elements of the slice at a time. The chunks are device
    /// slices and do not overlap. If `chunk_size` does not divide the length of the slice, then the
    /// last chunk will not have length `chunk_size`.
    ///
    /// See `exact_chunks` for a variant of this iterator that returns chunks of always exactly
    /// `chunk_size` elements.
    ///
    /// # Panics
    ///
    /// Panics if `chunk_size` is 0.
    ///
    /// # Examples
    ///
    /// ```
    /// # let _context = cust::quick_init().unwrap();
    /// use cust::memory::*;
    /// let slice = DeviceBuffer::from_slice(&[1u64, 2, 3, 4, 5]).unwrap();
    /// let mut iter = slice.chunks(2);
    ///
    /// assert_eq!(iter.next().unwrap().len(), 2);
    ///
    /// let mut host_buf = [0u64, 0];
    /// iter.next().unwrap().copy_to(&mut host_buf).unwrap();
    /// assert_eq!([3, 4], host_buf);
    ///
    /// assert_eq!(iter.next().unwrap().len(), 1);
    ///
    /// ```
    pub fn chunks(&self, chunk_size: usize) -> DeviceChunks<T> {
        DeviceChunks(self.0.chunks(chunk_size))
    }

    /// Returns an iterator over `chunk_size` elements of the slice at a time. The chunks are
    /// mutable device slices and do not overlap. If `chunk_size` does not divide the length of the
    /// slice, then the last chunk will not have length `chunk_size`.
    ///
    /// See `exact_chunks` for a variant of this iterator that returns chunks of always exactly
    /// `chunk_size` elements.
    ///
    /// # Panics
    ///
    /// Panics if `chunk_size` is 0.
    ///
    /// # Examples
    ///
    /// ```
    /// # let _context = cust::quick_init().unwrap();
    /// use cust::memory::*;
    /// let mut slice = DeviceBuffer::from_slice(&[0u64, 0, 0, 0, 0]).unwrap();
    /// {
    ///     let mut iter = slice.chunks_mut(2);
    ///
    ///     assert_eq!(iter.next().unwrap().len(), 2);
    ///
    ///     let host_buf = [2u64, 3];
    ///     iter.next().unwrap().copy_from(&host_buf).unwrap();
    ///
    ///     assert_eq!(iter.next().unwrap().len(), 1);
    /// }
    ///
    /// let mut host_buf = [0u64, 0, 0, 0, 0];
    /// slice.copy_to(&mut host_buf).unwrap();
    /// assert_eq!([0u64, 0, 2, 3, 0], host_buf);
    /// ```
    pub fn chunks_mut(&mut self, chunk_size: usize) -> DeviceChunksMut<T> {
        DeviceChunksMut(self.0.chunks_mut(chunk_size))
    }

    /// Private function used to transmute a CPU slice (which must have the device pointer as it's
    /// buffer pointer) to a DeviceSlice. Completely unsafe.
    pub(super) unsafe fn from_slice(slice: &[T]) -> &DSlice<T> {
        &*(slice as *const [T] as *const DSlice<T>)
    }

    /// Private function used to transmute a mutable CPU slice (which must have the device pointer
    /// as it's buffer pointer) to a mutable DeviceSlice. Completely unsafe.
    pub(super) unsafe fn from_slice_mut(slice: &mut [T]) -> &mut DSlice<T> {
        &mut *(slice as *mut [T] as *mut DSlice<T>)
    }

    /// Returns a `DevicePointer<T>` to the buffer.
    ///
    /// The caller must ensure that the buffer outlives the returned pointer, or it will end up
    /// pointing to garbage.
    ///
    /// Modifying `DeviceBuffer` is guaranteed not to cause its buffer to be reallocated, so pointers
    /// cannot be invalidated in that manner, but other types may be added in the future which can
    /// reallocate.
    pub fn as_device_ptr(&mut self) -> DevicePointer<T> {
        unsafe { DevicePointer::wrap(self.0.as_mut_ptr()) }
    }

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
    pub unsafe fn from_raw_parts<'a>(data: DevicePointer<T>, len: usize) -> &'a DSlice<T> {
        DSlice::from_slice(slice::from_raw_parts(data.as_raw(), len))
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
    pub unsafe fn from_raw_parts_mut<'a>(
        mut data: DevicePointer<T>,
        len: usize,
    ) -> &'a mut DSlice<T> {
        DSlice::from_slice_mut(slice::from_raw_parts_mut(data.as_raw_mut(), len))
    }
}

/// An iterator over a [`DeviceSlice`](struct.DeviceSlice.html) in (non-overlapping) chunks
/// (`chunk_size` elements at a time).
///
/// When the slice len is not evenly divided by the chunk size, the last slice of the iteration will
/// be the remainder.
///
/// This struct is created by the `chunks` method on `DeviceSlices`.
#[derive(Debug, Clone)]
pub struct DeviceChunks<'a, T: 'a>(Chunks<'a, T>);
impl<'a, T> Iterator for DeviceChunks<'a, T> {
    type Item = &'a DSlice<T>;

    fn next(&mut self) -> Option<&'a DSlice<T>> {
        self.0
            .next()
            .map(|slice| unsafe { DSlice::from_slice(slice) })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }

    fn count(self) -> usize {
        self.0.len()
    }

    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        self.0
            .nth(n)
            .map(|slice| unsafe { DSlice::from_slice(slice) })
    }

    #[inline]
    fn last(self) -> Option<Self::Item> {
        self.0
            .last()
            .map(|slice| unsafe { DSlice::from_slice(slice) })
    }
}
impl<'a, T> DoubleEndedIterator for DeviceChunks<'a, T> {
    #[inline]
    fn next_back(&mut self) -> Option<&'a DSlice<T>> {
        self.0
            .next_back()
            .map(|slice| unsafe { DSlice::from_slice(slice) })
    }
}
impl<'a, T> ExactSizeIterator for DeviceChunks<'a, T> {}
impl<'a, T> FusedIterator for DeviceChunks<'a, T> {}

/// An iterator over a [`DeviceSlice`](struct.DeviceSlice.html) in (non-overlapping) mutable chunks
/// (`chunk_size` elements at a time).
///
/// When the slice len is not evenly divided by the chunk size, the last slice of the iteration will
/// be the remainder.
///
/// This struct is created by the `chunks` method on `DeviceSlices`.
#[derive(Debug)]
pub struct DeviceChunksMut<'a, T: 'a>(ChunksMut<'a, T>);
impl<'a, T> Iterator for DeviceChunksMut<'a, T> {
    type Item = &'a mut DSlice<T>;

    fn next(&mut self) -> Option<&'a mut DSlice<T>> {
        self.0
            .next()
            .map(|slice| unsafe { DSlice::from_slice_mut(slice) })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }

    fn count(self) -> usize {
        self.0.len()
    }

    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        self.0
            .nth(n)
            .map(|slice| unsafe { DSlice::from_slice_mut(slice) })
    }

    #[inline]
    fn last(self) -> Option<Self::Item> {
        self.0
            .last()
            .map(|slice| unsafe { DSlice::from_slice_mut(slice) })
    }
}
impl<'a, T> DoubleEndedIterator for DeviceChunksMut<'a, T> {
    #[inline]
    fn next_back(&mut self) -> Option<&'a mut DSlice<T>> {
        self.0
            .next_back()
            .map(|slice| unsafe { DSlice::from_slice_mut(slice) })
    }
}
impl<'a, T> ExactSizeIterator for DeviceChunksMut<'a, T> {}
impl<'a, T> FusedIterator for DeviceChunksMut<'a, T> {}

macro_rules! impl_index {
    ($($t:ty)*) => {
        $(
            impl<T> Index<$t> for DSlice<T>
            {
                type Output = DSlice<T>;

                fn index(&self, index: $t) -> &Self {
                    unsafe { DSlice::from_slice(self.0.index(index)) }
                }
            }

            impl<T> IndexMut<$t> for DSlice<T>
            {
                fn index_mut(&mut self, index: $t) -> &mut Self {
                    unsafe { DSlice::from_slice_mut( self.0.index_mut(index)) }
                }
            }
        )*
    }
}
impl_index! {
    Range<usize>
    RangeFull
    RangeFrom<usize>
    RangeInclusive<usize>
    RangeTo<usize>
    RangeToInclusive<usize>
}
impl<T> crate::private::Sealed for DSlice<T> {}
impl<T: DeviceCopy, I: AsRef<[T]> + AsMut<[T]> + ?Sized> CopyDestination<I> for DSlice<T> {
    fn copy_from(&mut self, val: &I) -> CudaResult<()> {
        let val = val.as_ref();
        assert!(
            self.len() == val.len(),
            "destination and source slices have different lengths"
        );
        let size = mem::size_of::<T>() * self.len();
        if size != 0 {
            unsafe {
                cuda::cuMemcpyHtoD_v2(
                    self.0.as_mut_ptr() as u64,
                    val.as_ptr() as *const c_void,
                    size,
                )
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
                cuda::cuMemcpyDtoH_v2(val.as_mut_ptr() as *mut c_void, self.as_ptr() as u64, size)
                    .to_result()?
            }
        }
        Ok(())
    }
}
impl<T: DeviceCopy> CopyDestination<DSlice<T>> for DSlice<T> {
    fn copy_from(&mut self, val: &DSlice<T>) -> CudaResult<()> {
        assert!(
            self.len() == val.len(),
            "destination and source slices have different lengths"
        );
        let size = mem::size_of::<T>() * self.len();
        if size != 0 {
            unsafe {
                cuda::cuMemcpyDtoD_v2(self.0.as_mut_ptr() as u64, val.as_ptr() as u64, size)
                    .to_result()?
            }
        }
        Ok(())
    }

    fn copy_to(&self, val: &mut DSlice<T>) -> CudaResult<()> {
        assert!(
            self.len() == val.len(),
            "destination and source slices have different lengths"
        );
        let size = mem::size_of::<T>() * self.len();
        if size != 0 {
            unsafe {
                cuda::cuMemcpyDtoD_v2(val.as_mut_ptr() as u64, self.as_ptr() as u64, size)
                    .to_result()?
            }
        }
        Ok(())
    }
}
impl<T: DeviceCopy> CopyDestination<DBuffer<T>> for DSlice<T> {
    fn copy_from(&mut self, val: &DBuffer<T>) -> CudaResult<()> {
        self.copy_from(val as &DSlice<T>)
    }

    fn copy_to(&self, val: &mut DBuffer<T>) -> CudaResult<()> {
        self.copy_to(val as &mut DSlice<T>)
    }
}
impl<T: DeviceCopy, I: AsRef<[T]> + AsMut<[T]> + ?Sized> AsyncCopyDestination<I> for DSlice<T> {
    unsafe fn async_copy_from(&mut self, val: &I, stream: &Stream) -> CudaResult<()> {
        let val = val.as_ref();
        assert!(
            self.len() == val.len(),
            "destination and source slices have different lengths"
        );
        let size = mem::size_of::<T>() * self.len();
        if size != 0 {
            cuda::cuMemcpyHtoDAsync_v2(
                self.0.as_mut_ptr() as u64,
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
                self.as_ptr() as u64,
                size,
                stream.as_inner(),
            )
            .to_result()?
        }
        Ok(())
    }
}
impl<T: DeviceCopy> AsyncCopyDestination<DSlice<T>> for DSlice<T> {
    unsafe fn async_copy_from(&mut self, val: &DSlice<T>, stream: &Stream) -> CudaResult<()> {
        assert!(
            self.len() == val.len(),
            "destination and source slices have different lengths"
        );
        let size = mem::size_of::<T>() * self.len();
        if size != 0 {
            cuda::cuMemcpyDtoDAsync_v2(
                self.0.as_mut_ptr() as u64,
                val.as_ptr() as u64,
                size,
                stream.as_inner(),
            )
            .to_result()?
        }
        Ok(())
    }

    unsafe fn async_copy_to(&self, val: &mut DSlice<T>, stream: &Stream) -> CudaResult<()> {
        assert!(
            self.len() == val.len(),
            "destination and source slices have different lengths"
        );
        let size = mem::size_of::<T>() * self.len();
        if size != 0 {
            cuda::cuMemcpyDtoDAsync_v2(
                val.as_mut_ptr() as u64,
                self.as_ptr() as u64,
                size,
                stream.as_inner(),
            )
            .to_result()?
        }
        Ok(())
    }
}
impl<T: DeviceCopy> AsyncCopyDestination<DBuffer<T>> for DSlice<T> {
    unsafe fn async_copy_from(&mut self, val: &DBuffer<T>, stream: &Stream) -> CudaResult<()> {
        self.async_copy_from(val as &DSlice<T>, stream)
    }

    unsafe fn async_copy_to(&self, val: &mut DBuffer<T>, stream: &Stream) -> CudaResult<()> {
        self.async_copy_to(val as &mut DSlice<T>, stream)
    }
}
