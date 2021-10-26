use crate::error::{CudaResult, DropResult, ToResult};
use crate::memory::device::AsyncCopyDestination;
use crate::memory::device::CopyDestination;
use crate::memory::malloc::{cuda_free, cuda_malloc};
use crate::memory::DeviceCopy;
use crate::memory::DevicePointer;
use crate::stream::Stream;
use crate::sys as cuda;
use std::fmt::{self, Pointer};
use std::mem;

use std::os::raw::c_void;

/// A pointer type for heap-allocation in CUDA device memory.
///
/// See the [`module-level documentation`](../memory/index.html) for more information on device memory.
#[derive(Debug)]
pub struct DBox<T> {
    pub(crate) ptr: DevicePointer<T>,
}
impl<T: DeviceCopy> DBox<T> {
    /// Allocate device memory and place val into it.
    ///
    /// This doesn't actually allocate if `T` is zero-sized.
    ///
    /// # Errors
    ///
    /// If a CUDA error occurs, return the error.
    ///
    /// # Examples
    ///
    /// ```
    /// # let _context = cust::quick_init().unwrap();
    /// use cust::memory::*;
    /// let five = DBox::new(&5).unwrap();
    /// ```
    pub fn new(val: &T) -> CudaResult<Self> {
        let mut dev_box = unsafe { DBox::uninitialized()? };
        dev_box.copy_from(val)?;
        Ok(dev_box)
    }
}

impl<T: DeviceCopy + Default> DBox<T> {
    /// Read the data back from the GPU into host memory.
    pub fn as_host_value(&self) -> CudaResult<T> {
        let mut val = T::default();
        self.copy_to(&mut val)?;
        Ok(val)
    }
}

impl<T> DBox<T> {
    /// Allocate device memory, but do not initialize it.
    ///
    /// This doesn't actually allocate if `T` is zero-sized.
    ///
    /// # Safety
    ///
    /// Since the backing memory is not initialized, this function is not safe. The caller must
    /// ensure that the backing memory is set to a valid value before it is read, else undefined
    /// behavior may occur.
    ///
    /// # Examples
    ///
    /// ```
    /// # let _context = cust::quick_init().unwrap();
    /// use cust::memory::*;
    /// let mut five = unsafe { DBox::uninitialized().unwrap() };
    /// five.copy_from(&5u64).unwrap();
    /// ```
    pub unsafe fn uninitialized() -> CudaResult<Self> {
        if mem::size_of::<T>() == 0 {
            Ok(DBox {
                ptr: DevicePointer::null(),
            })
        } else {
            let ptr = cuda_malloc(1)?;
            Ok(DBox { ptr })
        }
    }

    /// Allocate device memory and fill it with zeroes (`0u8`).
    ///
    /// This doesn't actually allocate if `T` is zero-sized.
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
    /// let mut zero = unsafe { DBox::zeroed().unwrap() };
    /// let mut value = 5u64;
    /// zero.copy_to(&mut value).unwrap();
    /// assert_eq!(0, value);
    /// ```
    pub unsafe fn zeroed() -> CudaResult<Self> {
        let mut new_box = DBox::uninitialized()?;
        if mem::size_of::<T>() != 0 {
            cuda::cuMemsetD8_v2(
                new_box.as_device_ptr().as_raw_mut() as u64,
                0,
                mem::size_of::<T>(),
            )
            .to_result()?;
        }
        Ok(new_box)
    }

    /// Constructs a DBox from a raw pointer.
    ///
    /// After calling this function, the raw pointer and the memory it points to is owned by the
    /// DBox. The DBox destructor will free the allocated memory, but will not call the destructor
    /// of `T`. This function may accept any pointer produced by the `cuMemAllocManaged` CUDA API
    /// call.
    ///
    /// # Safety
    ///
    /// This function is unsafe because improper use may lead to memory problems. For example, a
    /// double free may occur if this function is called twice on the same pointer, or a segfault
    /// may occur if the pointer is not one returned by the appropriate API call.
    ///
    /// # Examples
    ///
    /// ```
    /// # let _context = cust::quick_init().unwrap();
    /// use cust::memory::*;
    /// let x = DBox::new(&5).unwrap();
    /// let ptr = DBox::into_device(x).as_raw_mut();
    /// let x = unsafe { DBox::from_raw(ptr) };
    /// ```
    pub unsafe fn from_raw(ptr: *mut T) -> Self {
        DBox {
            ptr: DevicePointer::wrap(ptr),
        }
    }

    /// Constructs a DBox from a DevicePointer.
    ///
    /// After calling this function, the pointer and the memory it points to is owned by the
    /// DBox. The DBox destructor will free the allocated memory, but will not call the destructor
    /// of `T`. This function may accept any pointer produced by the `cuMemAllocManaged` CUDA API
    /// call, such as one taken from `DBox::into_device`.
    ///
    /// # Safety
    ///
    /// This function is unsafe because improper use may lead to memory problems. For example, a
    /// double free may occur if this function is called twice on the same pointer, or a segfault
    /// may occur if the pointer is not one returned by the appropriate API call.
    ///
    /// # Examples
    ///
    /// ```
    /// # let _context = cust::quick_init().unwrap();
    /// use cust::memory::*;
    /// let x = DBox::new(&5).unwrap();
    /// let ptr = DBox::into_device(x);
    /// let x = unsafe { DBox::from_device(ptr) };
    /// ```
    pub unsafe fn from_device(ptr: DevicePointer<T>) -> Self {
        DBox { ptr }
    }

    /// Consumes the DBox, returning the wrapped DevicePointer.
    ///
    /// After calling this function, the caller is responsible for the memory previously managed by
    /// the DBox. In particular, the caller should properly destroy T and deallocate the memory.
    /// The easiest way to do so is to create a new DBox using the `DBox::from_device` function.
    ///
    /// Note: This is an associated function, which means that you have to all it as
    /// `DBox::into_device(b)` instead of `b.into_device()` This is so that there is no conflict with
    /// a method on the inner type.
    ///
    /// # Examples
    ///
    /// ```
    /// # let _context = cust::quick_init().unwrap();
    /// use cust::memory::*;
    /// let x = DBox::new(&5).unwrap();
    /// let ptr = DBox::into_device(x);
    /// # unsafe { DBox::from_device(ptr) };
    /// ```
    #[allow(clippy::wrong_self_convention)]
    pub fn into_device(mut b: DBox<T>) -> DevicePointer<T> {
        let ptr = mem::replace(&mut b.ptr, DevicePointer::null());
        mem::forget(b);
        ptr
    }

    /// Returns the contained device pointer without consuming the box.
    ///
    /// This is useful for passing the box to a kernel launch.
    ///
    /// # Examples
    ///
    /// ```
    /// # let _context = cust::quick_init().unwrap();
    /// use cust::memory::*;
    /// let mut x = DBox::new(&5).unwrap();
    /// let ptr = x.as_device_ptr();
    /// println!("{:p}", ptr);
    /// ```
    pub fn as_device_ptr(&mut self) -> DevicePointer<T> {
        self.ptr
    }

    /// Destroy a `DBox`, returning an error.
    ///
    /// Deallocating device memory can return errors from previous asynchronous work. This function
    /// destroys the given box and returns the error and the un-destroyed box on failure.
    ///
    /// # Example
    ///
    /// ```
    /// # let _context = cust::quick_init().unwrap();
    /// use cust::memory::*;
    /// let x = DBox::new(&5).unwrap();
    /// match DBox::drop(x) {
    ///     Ok(()) => println!("Successfully destroyed"),
    ///     Err((e, dev_box)) => {
    ///         println!("Failed to destroy box: {:?}", e);
    ///         // Do something with dev_box
    ///     },
    /// }
    /// ```
    pub fn drop(mut dev_box: DBox<T>) -> DropResult<DBox<T>> {
        if dev_box.ptr.is_null() {
            return Ok(());
        }

        let ptr = mem::replace(&mut dev_box.ptr, DevicePointer::null());
        unsafe {
            match cuda_free(ptr) {
                Ok(()) => {
                    mem::forget(dev_box);
                    Ok(())
                }
                Err(e) => Err((e, DBox { ptr })),
            }
        }
    }
}
impl<T> Drop for DBox<T> {
    fn drop(&mut self) {
        if self.ptr.is_null() {
            return;
        }

        let ptr = mem::replace(&mut self.ptr, DevicePointer::null());
        unsafe {
            let _ = cuda_free(ptr);
        }
    }
}
impl<T> Pointer for DBox<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Pointer::fmt(&self.ptr, f)
    }
}
impl<T> crate::private::Sealed for DBox<T> {}
impl<T: DeviceCopy> CopyDestination<T> for DBox<T> {
    fn copy_from(&mut self, val: &T) -> CudaResult<()> {
        let size = mem::size_of::<T>();
        if size != 0 {
            unsafe {
                cuda::cuMemcpyHtoD_v2(
                    self.ptr.as_raw_mut() as u64,
                    val as *const T as *const c_void,
                    size,
                )
                .to_result()?
            }
        }
        Ok(())
    }

    fn copy_to(&self, val: &mut T) -> CudaResult<()> {
        let size = mem::size_of::<T>();
        if size != 0 {
            unsafe {
                cuda::cuMemcpyDtoH_v2(
                    val as *const T as *mut c_void,
                    self.ptr.as_raw() as u64,
                    size,
                )
                .to_result()?
            }
        }
        Ok(())
    }
}
impl<T: DeviceCopy> CopyDestination<DBox<T>> for DBox<T> {
    fn copy_from(&mut self, val: &DBox<T>) -> CudaResult<()> {
        let size = mem::size_of::<T>();
        if size != 0 {
            unsafe {
                cuda::cuMemcpyDtoD_v2(self.ptr.as_raw_mut() as u64, val.ptr.as_raw() as u64, size)
                    .to_result()?
            }
        }
        Ok(())
    }

    fn copy_to(&self, val: &mut DBox<T>) -> CudaResult<()> {
        let size = mem::size_of::<T>();
        if size != 0 {
            unsafe {
                cuda::cuMemcpyDtoD_v2(val.ptr.as_raw_mut() as u64, self.ptr.as_raw() as u64, size)
                    .to_result()?
            }
        }
        Ok(())
    }
}
impl<T: DeviceCopy> AsyncCopyDestination<DBox<T>> for DBox<T> {
    unsafe fn async_copy_from(&mut self, val: &DBox<T>, stream: &Stream) -> CudaResult<()> {
        let size = mem::size_of::<T>();
        if size != 0 {
            cuda::cuMemcpyDtoDAsync_v2(
                self.ptr.as_raw_mut() as u64,
                val.ptr.as_raw() as u64,
                size,
                stream.as_inner(),
            )
            .to_result()?
        }
        Ok(())
    }

    unsafe fn async_copy_to(&self, val: &mut DBox<T>, stream: &Stream) -> CudaResult<()> {
        let size = mem::size_of::<T>();
        if size != 0 {
            cuda::cuMemcpyDtoDAsync_v2(
                val.ptr.as_raw_mut() as u64,
                self.ptr.as_raw() as u64,
                size,
                stream.as_inner(),
            )
            .to_result()?
        }
        Ok(())
    }
}

#[cfg(test)]
mod test_device_box {
    use super::*;

    #[derive(Clone, Copy, Debug)]
    struct ZeroSizedType;
    unsafe impl DeviceCopy for ZeroSizedType {}

    #[test]
    fn test_allocate_and_free_device_box() {
        let _context = crate::quick_init().unwrap();
        let x = DBox::new(&5u64).unwrap();
        drop(x);
    }

    #[test]
    fn test_device_box_allocates_for_non_zst() {
        let _context = crate::quick_init().unwrap();
        let x = DBox::new(&5u64).unwrap();
        let ptr = DBox::into_device(x);
        assert!(!ptr.is_null());
        let _ = unsafe { DBox::from_device(ptr) };
    }

    #[test]
    fn test_device_box_doesnt_allocate_for_zero_sized_type() {
        let _context = crate::quick_init().unwrap();
        let x = DBox::new(&ZeroSizedType).unwrap();
        let ptr = DBox::into_device(x);
        assert!(ptr.is_null());
        let _ = unsafe { DBox::from_device(ptr) };
    }

    #[test]
    fn test_into_from_device() {
        let _context = crate::quick_init().unwrap();
        let x = DBox::new(&5u64).unwrap();
        let ptr = DBox::into_device(x);
        let _ = unsafe { DBox::from_device(ptr) };
    }

    #[test]
    fn test_copy_host_to_device() {
        let _context = crate::quick_init().unwrap();
        let y = 5u64;
        let mut x = DBox::new(&0u64).unwrap();
        x.copy_from(&y).unwrap();
        let mut z = 10u64;
        x.copy_to(&mut z).unwrap();
        assert_eq!(y, z);
    }

    #[test]
    fn test_copy_device_to_host() {
        let _context = crate::quick_init().unwrap();
        let x = DBox::new(&5u64).unwrap();
        let mut y = 0u64;
        x.copy_to(&mut y).unwrap();
        assert_eq!(5, y);
    }

    #[test]
    fn test_copy_device_to_device() {
        let _context = crate::quick_init().unwrap();
        let x = DBox::new(&5u64).unwrap();
        let mut y = DBox::new(&0u64).unwrap();
        let mut z = DBox::new(&0u64).unwrap();
        x.copy_to(&mut y).unwrap();
        z.copy_from(&y).unwrap();

        let mut h = 0u64;
        z.copy_to(&mut h).unwrap();
        assert_eq!(5, h);
    }

    #[test]
    fn test_device_pointer_implements_traits_safely() {
        let _context = crate::quick_init().unwrap();
        let mut x = DBox::new(&5u64).unwrap();
        let mut y = DBox::new(&0u64).unwrap();

        // If the impls dereference the pointer, this should segfault.
        let _ = Ord::cmp(&x.as_device_ptr(), &y.as_device_ptr());
        let _ = PartialOrd::partial_cmp(&x.as_device_ptr(), &y.as_device_ptr());
        let _ = PartialEq::eq(&x.as_device_ptr(), &y.as_device_ptr());

        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        std::hash::Hash::hash(&x.as_device_ptr(), &mut hasher);

        let _ = format!("{:?}", x.as_device_ptr());
        let _ = format!("{:p}", x.as_device_ptr());
    }
}
