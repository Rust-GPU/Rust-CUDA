use super::DeviceCopy;
use crate::error::*;
use crate::memory::DevicePointer;
use crate::memory::UnifiedPointer;
use crate::sys as cuda;
use std::mem;
use std::os::raw::c_void;
use std::ptr;

/// Unsafe wrapper around the `cuMemAlloc` function, which allocates some device memory and
/// returns a [`DevicePointer`](struct.DevicePointer.html) pointing to it. The memory is not cleared.
///
/// Note that `count` is in units of T; thus a `count` of 3 will allocate `3 * size_of::<T>()` bytes
/// of memory.
///
/// Memory buffers allocated using `cuda_malloc` must be freed using [`cuda_free`](fn.cuda_free.html).
///
/// # Errors
///
/// If allocating memory fails, returns the CUDA error value.
/// If the number of bytes to allocate is zero (either because count is zero or because T is a
/// zero-sized type), or if the size of the allocation would overflow a usize, returns InvalidValue.
///
/// # Safety
///
/// Since the allocated memory is not initialized, the caller must ensure that it is initialized
/// before copying it to the host in any way. Additionally, the caller must ensure that the memory
/// allocated is freed using cuda_free, or the memory will be leaked.
///
/// # Examples
///
/// ```
/// # let _context = cust::quick_init().unwrap();
/// use cust::memory::*;
/// unsafe {
///     // Allocate space for 5 u64s
///     let device_buffer = cuda_malloc::<u64>(5).unwrap();
///     cuda_free(device_buffer).unwrap();
/// }
/// ```
pub unsafe fn cuda_malloc<T>(count: usize) -> CudaResult<DevicePointer<T>> {
    let size = count.checked_mul(mem::size_of::<T>()).unwrap_or(0);
    if size == 0 {
        return Err(CudaError::InvalidMemoryAllocation);
    }

    let mut ptr: *mut c_void = ptr::null_mut();
    cuda::cuMemAlloc_v2(&mut ptr as *mut *mut c_void as *mut u64, size).to_result()?;
    let ptr = ptr as *mut T;
    Ok(DevicePointer::wrap(ptr as *mut T))
}

/// Unsafe wrapper around the `cuMemAllocManaged` function, which allocates some unified memory and
/// returns a [`UnifiedPointer`](struct.UnifiedPointer.html) pointing to it. The memory is not cleared.
///
/// Note that `count` is in units of T; thus a `count` of 3 will allocate `3 * size_of::<T>()` bytes
/// of memory.
///
/// Memory buffers allocated using `cuda_malloc_unified` must be freed using [`cuda_free`](fn.cuda_free.html).
///
/// # Errors
///
/// If allocating memory fails, returns the CUDA error value.
/// If the number of bytes to allocate is zero (either because count is zero or because T is a
/// zero-sized type), or if the size of the allocation would overflow a usize, returns InvalidValue.
///
/// # Safety
///
/// Since the allocated memory is not initialized, the caller must ensure that it is initialized
/// before reading from it in any way. Additionally, the caller must ensure that the memory
/// allocated is freed using cuda_free, or the memory will be leaked.
///
/// # Examples
///
/// ```
/// # let _context = cust::quick_init().unwrap();
/// use cust::memory::*;
/// unsafe {
///     // Allocate space for a u64
///     let mut unified_buffer = cuda_malloc_unified::<u64>(1).unwrap();
///     // Write to it
///     *unified_buffer.as_raw_mut() = 5u64;
///     cuda_free_unified(unified_buffer).unwrap();
/// }
/// ```
pub unsafe fn cuda_malloc_unified<T: DeviceCopy>(count: usize) -> CudaResult<UnifiedPointer<T>> {
    let size = count.checked_mul(mem::size_of::<T>()).unwrap_or(0);
    if size == 0 {
        return Err(CudaError::InvalidMemoryAllocation);
    }

    let mut ptr: *mut c_void = ptr::null_mut();
    cuda::cuMemAllocManaged(
        &mut ptr as *mut *mut c_void as *mut u64,
        size,
        cuda::CUmemAttach_flags_enum::CU_MEM_ATTACH_GLOBAL as u32,
    )
    .to_result()?;
    let ptr = ptr as *mut T;
    Ok(UnifiedPointer::wrap(ptr as *mut T))
}

/// Free memory allocated with [`cuda_malloc`](fn.cuda_malloc.html).
///
/// # Errors
///
/// If freeing memory fails, returns the CUDA error value. If the given pointer is null, returns
/// InvalidValue.
///
/// # Safety
///
/// The given pointer must have been allocated with `cuda_malloc`, or null.
/// The caller is responsible for ensuring that no other pointers to the deallocated buffer exist.
///
/// # Examples
///
/// ```
/// # let _context = cust::quick_init().unwrap();
/// use cust::memory::*;
/// unsafe {
///     let device_buffer = cuda_malloc::<u64>(5).unwrap();
///     // Free allocated memory.
///     cuda_free(device_buffer).unwrap();
/// }
/// ```
pub unsafe fn cuda_free<T>(mut p: DevicePointer<T>) -> CudaResult<()> {
    let ptr = p.as_raw_mut();
    if ptr.is_null() {
        return Err(CudaError::InvalidMemoryAllocation);
    }

    cuda::cuMemFree_v2(ptr as u64).to_result()?;
    Ok(())
}

/// Free memory allocated with [`cuda_malloc_unified`](fn.cuda_malloc_unified.html).
///
/// # Errors
///
/// If freeing memory fails, returns the CUDA error value. If the given pointer is null, returns
/// InvalidValue.
///
/// # Safety
///
/// The given pointer must have been allocated with `cuda_malloc_unified`, or null.
/// The caller is responsible for ensuring that no other pointers to the deallocated buffer exist.
///
/// # Examples
///
/// ```
/// # let _context = cust::quick_init().unwrap();
/// use cust::memory::*;
/// unsafe {
///     let unified_buffer = cuda_malloc_unified::<u64>(5).unwrap();
///     // Free allocated memory.
///     cuda_free_unified(unified_buffer).unwrap();
/// }
/// ```
pub unsafe fn cuda_free_unified<T: DeviceCopy>(mut p: UnifiedPointer<T>) -> CudaResult<()> {
    let ptr = p.as_raw_mut();
    if ptr.is_null() {
        return Err(CudaError::InvalidMemoryAllocation);
    }

    cuda::cuMemFree_v2(ptr as u64).to_result()?;
    Ok(())
}

/// Unsafe wrapper around the `cuMemAllocHost` function, which allocates some page-locked host memory
/// and returns a raw pointer pointing to it. The memory is not cleared.
///
/// Note that `count` is in units of T; thus a `count` of 3 will allocate `3 * size_of::<T>()` bytes
/// of memory.
///
/// Memory buffers allocated using `cuda_malloc_locked` must be freed using [`cuda_free_locked`](fn.cuda_free_locked.html).
///
/// # Errors
///
/// If allocating memory fails, returns the CUDA error value.
/// If the number of bytes to allocate is zero (either because count is zero or because T is a
/// zero-sized type), or if the size of the allocation would overflow a usize, returns InvalidValue.
///
/// # Safety
///
/// Since the allocated memory is not initialized, the caller must ensure that it is initialized
/// before reading from it in any way. Additionally, the caller must ensure that the memory
/// allocated is freed using `cuda_free_locked`, or the memory will be leaked.
///
/// # Examples
///
/// ```
/// # let _context = cust::quick_init().unwrap();
/// use cust::memory::*;
/// unsafe {
///     // Allocate space for 5 u64s
///     let locked_buffer = cuda_malloc_locked::<u64>(5).unwrap();
///     cuda_free_locked(locked_buffer).unwrap();
/// }
/// ```
pub unsafe fn cuda_malloc_locked<T>(count: usize) -> CudaResult<*mut T> {
    let size = count.checked_mul(mem::size_of::<T>()).unwrap_or(0);
    if size == 0 {
        return Err(CudaError::InvalidMemoryAllocation);
    }

    let mut ptr: *mut c_void = ptr::null_mut();
    cuda::cuMemAllocHost_v2(&mut ptr as *mut *mut c_void, size).to_result()?;
    let ptr = ptr as *mut T;
    Ok(ptr as *mut T)
}

/// Free page-locked memory allocated with [`cuda_malloc_host`](fn.cuda_malloc_host.html).
///
/// # Errors
///
/// If freeing memory fails, returns the CUDA error value. If the given pointer is null, returns
/// InvalidValue.
///
/// # Safety
///
/// The given pointer must have been allocated with `cuda_malloc_locked`, or null.
/// The caller is responsible for ensuring that no other pointers to the deallocated buffer exist.
///
/// # Examples
///
/// ```
/// # let _context = cust::quick_init().unwrap();
/// use cust::memory::*;
/// unsafe {
///     let locked_buffer = cuda_malloc_locked::<u64>(5).unwrap();
///     // Free allocated memory
///     cuda_free_locked(locked_buffer).unwrap();
/// }
/// ```
pub unsafe fn cuda_free_locked<T>(ptr: *mut T) -> CudaResult<()> {
    if ptr.is_null() {
        return Err(CudaError::InvalidMemoryAllocation);
    }

    cuda::cuMemFreeHost(ptr as *mut c_void).to_result()?;
    Ok(())
}

#[cfg(test)]
mod test {
    use super::*;

    #[derive(Clone, Copy, Debug)]
    struct ZeroSizedType;
    unsafe impl DeviceCopy for ZeroSizedType {}

    #[test]
    fn test_cuda_malloc() {
        let _context = crate::quick_init().unwrap();
        unsafe {
            let device_mem = cuda_malloc::<u64>(1).unwrap();
            assert!(!device_mem.is_null());
            cuda_free(device_mem).unwrap();
        }
    }

    #[test]
    fn test_cuda_malloc_zero_bytes() {
        let _context = crate::quick_init().unwrap();
        unsafe {
            assert_eq!(
                CudaError::InvalidMemoryAllocation,
                cuda_malloc::<u64>(0).unwrap_err()
            );
        }
    }

    #[test]
    fn test_cuda_malloc_zero_sized() {
        let _context = crate::quick_init().unwrap();
        unsafe {
            assert_eq!(
                CudaError::InvalidMemoryAllocation,
                cuda_malloc::<ZeroSizedType>(10).unwrap_err()
            );
        }
    }

    #[test]
    fn test_cuda_alloc_overflow() {
        let _context = crate::quick_init().unwrap();
        unsafe {
            assert_eq!(
                CudaError::InvalidMemoryAllocation,
                cuda_malloc::<u64>(::std::usize::MAX - 1).unwrap_err()
            );
        }
    }

    #[test]
    fn test_cuda_malloc_unified() {
        let _context = crate::quick_init().unwrap();
        unsafe {
            let mut unified = cuda_malloc_unified::<u64>(1).unwrap();
            assert!(!unified.is_null());

            // Write to the allocated memory
            *unified.as_raw_mut() = 64;

            cuda_free_unified(unified).unwrap();
        }
    }

    #[test]
    fn test_cuda_malloc_unified_zero_bytes() {
        let _context = crate::quick_init().unwrap();
        unsafe {
            assert_eq!(
                CudaError::InvalidMemoryAllocation,
                cuda_malloc_unified::<u64>(0).unwrap_err()
            );
        }
    }

    #[test]
    fn test_cuda_malloc_unified_zero_sized() {
        let _context = crate::quick_init().unwrap();
        unsafe {
            assert_eq!(
                CudaError::InvalidMemoryAllocation,
                cuda_malloc_unified::<ZeroSizedType>(10).unwrap_err()
            );
        }
    }

    #[test]
    fn test_cuda_malloc_unified_overflow() {
        let _context = crate::quick_init().unwrap();
        unsafe {
            assert_eq!(
                CudaError::InvalidMemoryAllocation,
                cuda_malloc_unified::<u64>(::std::usize::MAX - 1).unwrap_err()
            );
        }
    }

    #[test]
    fn test_cuda_free_null() {
        let _context = crate::quick_init().unwrap();
        let null = ::std::ptr::null_mut::<u64>();
        unsafe {
            assert_eq!(
                CudaError::InvalidMemoryAllocation,
                cuda_free(DevicePointer::wrap(null)).unwrap_err()
            );
        }
    }

    #[test]
    fn test_cuda_malloc_locked() {
        let _context = crate::quick_init().unwrap();
        unsafe {
            let locked = cuda_malloc_locked::<u64>(1).unwrap();
            assert!(!locked.is_null());

            // Write to the allocated memory
            *locked = 64;

            cuda_free_locked(locked).unwrap();
        }
    }

    #[test]
    fn test_cuda_malloc_locked_zero_bytes() {
        let _context = crate::quick_init().unwrap();
        unsafe {
            assert_eq!(
                CudaError::InvalidMemoryAllocation,
                cuda_malloc_locked::<u64>(0).unwrap_err()
            );
        }
    }

    #[test]
    fn test_cuda_malloc_locked_zero_sized() {
        let _context = crate::quick_init().unwrap();
        unsafe {
            assert_eq!(
                CudaError::InvalidMemoryAllocation,
                cuda_malloc_locked::<ZeroSizedType>(10).unwrap_err()
            );
        }
    }

    #[test]
    fn test_cuda_malloc_locked_overflow() {
        let _context = crate::quick_init().unwrap();
        unsafe {
            assert_eq!(
                CudaError::InvalidMemoryAllocation,
                cuda_malloc_locked::<u64>(::std::usize::MAX - 1).unwrap_err()
            );
        }
    }

    #[test]
    fn test_cuda_free_locked_null() {
        let _context = crate::quick_init().unwrap();
        unsafe {
            assert_eq!(
                CudaError::InvalidMemoryAllocation,
                cuda_free_locked(::std::ptr::null_mut::<u64>()).unwrap_err()
            );
        }
    }
}
