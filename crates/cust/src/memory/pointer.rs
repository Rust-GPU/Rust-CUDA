use crate::memory::DeviceCopy;

use core::{
    cmp::Ordering,
    fmt::{self, Debug, Pointer},
    hash::{Hash, Hasher},
    ptr,
};

macro_rules! derive_traits {
    ( $( $Ptr:ty )* ) => ($(
        impl<T: ?Sized> Debug for $Ptr {
            fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                Debug::fmt(&self.0, f)
            }
        }
        impl<T: ?Sized> Pointer for $Ptr {
            fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                Pointer::fmt(&self.0, f)
            }
        }

        impl<T: ?Sized> Hash for $Ptr {
            fn hash<H: Hasher>(&self, h: &mut H) {
                Hash::hash(&self.0, h);
            }
        }

        impl<T: ?Sized> PartialEq for $Ptr {
            fn eq(&self, other: &$Ptr) -> bool {
                PartialEq::eq(&self.0, &other.0)
            }
        }

        impl<T: ?Sized> Eq for $Ptr {}

        impl<T: ?Sized> PartialOrd for $Ptr {
            fn partial_cmp(&self, other: &$Ptr) -> Option<Ordering> {
                PartialOrd::partial_cmp(&self.0, &other.0)
            }
        }

        impl<T: ?Sized> Ord for $Ptr {
            fn cmp(&self, other: &$Ptr) -> Ordering {
                Ord::cmp(&self.0, &other.0)
            }
        }

        impl<T: ?Sized> Clone for $Ptr {
            fn clone(&self) -> Self {
                Self(self.0)
            }
        }
        impl<T: ?Sized> Copy for $Ptr {}
    )*)
}
derive_traits!(DevicePointer<T> UnifiedPointer<T>);

/// A pointer to device memory.
///
/// `DevicePointer` cannot be dereferenced by the CPU, as it is a pointer to a memory allocation in
/// the device. It can be safely copied to the device (eg. as part of a kernel launch) and either
/// unwrapped or transmuted to an appropriate pointer.
///
/// `DevicePointer` is guaranteed to have an equivalent internal representation to a raw pointer.
/// Thus, it can be safely reinterpreted or transmuted to `*mut T`. It is safe to pass a
/// `DevicePointer` through an FFI boundary to C code expecting a `*mut T`, so long as the code on
/// the other side of that boundary does not attempt to dereference the pointer on the CPU. It is
/// thus possible to pass a `DevicePointer` to a CUDA kernel written in C.
#[repr(transparent)]
pub struct DevicePointer<T: ?Sized>(*mut T);

unsafe impl<T: ?Sized> DeviceCopy for DevicePointer<T> {}

impl<T: ?Sized> DevicePointer<T> {
    /// Wrap the given raw pointer in a DevicePointer. The given pointer is assumed to be a valid,
    /// device pointer or null.
    ///
    /// # Safety
    ///
    /// The given pointer must have been allocated with [`cuda_malloc`](fn.cuda_malloc.html) or
    /// be null.
    ///
    /// # Examples
    ///
    /// ```
    /// # let _context = cust::quick_init().unwrap();
    /// use cust::memory::*;
    /// use std::ptr;
    /// unsafe {
    ///     let null : *mut u64 = ptr::null_mut();
    ///     assert!(DevicePointer::wrap(null).is_null());
    /// }
    /// ```
    pub unsafe fn wrap(ptr: *mut T) -> Self {
        DevicePointer(ptr)
    }

    /// Returns the contained pointer as a raw pointer. The returned pointer is not valid on the CPU
    /// and must not be dereferenced.
    ///
    /// # Examples
    ///
    /// ```
    /// # let _context = cust::quick_init().unwrap();
    /// use cust::memory::*;
    /// unsafe {
    ///     let dev_ptr = cuda_malloc::<u64>(1).unwrap();
    ///     let ptr: *const u64 = dev_ptr.as_raw();
    ///     cuda_free(dev_ptr);
    /// }
    /// ```
    pub fn as_raw(self) -> *const T {
        self.0
    }

    /// Returns the contained pointer as a mutable raw pointer. The returned pointer is not valid on the CPU
    /// and must not be dereferenced.
    ///
    /// # Examples
    ///
    /// ```
    /// # let _context = cust::quick_init().unwrap();
    /// use cust::memory::*;
    /// unsafe {
    ///     let mut dev_ptr = cuda_malloc::<u64>(1).unwrap();
    ///     let ptr: *mut u64 = dev_ptr.as_raw_mut();
    ///     cuda_free(dev_ptr);
    /// }
    /// ```
    pub fn as_raw_mut(&mut self) -> *mut T {
        self.0
    }

    /// Returns true if the pointer is null.
    /// # Examples
    ///
    /// ```
    /// # let _context = cust::quick_init().unwrap();
    /// use cust::memory::*;
    /// use std::ptr;
    /// unsafe {
    ///     let null : *mut u64 = ptr::null_mut();
    ///     assert!(DevicePointer::wrap(null).is_null());
    /// }
    /// ```
    pub fn is_null(self) -> bool {
        self.0.is_null()
    }

    /// Returns a null device pointer.
    ///
    /// # Examples:
    ///
    /// ```
    /// # let _context = cust::quick_init().unwrap();
    /// use cust::memory::*;
    /// let ptr : DevicePointer<u64> = DevicePointer::null();
    /// assert!(ptr.is_null());
    /// ```
    pub fn null() -> Self
    where
        T: Sized,
    {
        unsafe { Self::wrap(ptr::null_mut()) }
    }

    /// Calculates the offset from a device pointer.
    ///
    /// `count` is in units of T; eg. a `count` of 3 represents a pointer offset of
    /// `3 * size_of::<T>()` bytes.
    ///
    /// # Safety
    ///
    /// If any of the following conditions are violated, the result is Undefined
    /// Behavior:
    ///
    /// * Both the starting and resulting pointer must be either in bounds or one
    ///   byte past the end of *the same* allocated object.
    ///
    /// * The computed offset, **in bytes**, cannot overflow an `isize`.
    ///
    /// * The offset being in bounds cannot rely on "wrapping around" the address
    ///   space. That is, the infinite-precision sum, **in bytes** must fit in a usize.
    ///
    /// Consider using `wrapping_offset` instead if these constraints are
    /// difficult to satisfy. The only advantage of this method is that it
    /// enables more aggressive compiler optimizations.
    ///
    /// # Examples
    ///
    /// ```
    /// # let _context = cust::quick_init().unwrap();
    /// use cust::memory::*;
    /// unsafe {
    ///     let mut dev_ptr = cuda_malloc::<u64>(5).unwrap();
    ///     let offset = dev_ptr.offset(1); // Points to the 2nd u64 in the buffer
    ///     cuda_free(dev_ptr); // Must free the buffer using the original pointer
    /// }
    /// ```
    pub unsafe fn offset(self, count: isize) -> Self
    where
        T: Sized,
    {
        Self::wrap(self.0.offset(count))
    }

    /// Calculates the offset from a device pointer using wrapping arithmetic.
    ///
    /// `count` is in units of T; eg. a `count` of 3 represents a pointer offset of
    /// `3 * size_of::<T>()` bytes.
    ///
    /// # Safety
    ///
    /// The resulting pointer does not need to be in bounds, but it is
    /// potentially hazardous to dereference (which requires `unsafe`).
    /// In particular, the resulting pointer may *not* be used to access a
    /// different allocated object than the one `self` points to. In other
    /// words, `x.wrapping_offset(y.wrapping_offset_from(x))` is
    /// *not* the same as `y`, and dereferencing it is undefined behavior
    /// unless `x` and `y` point into the same allocated object.
    ///
    /// Always use `.offset(count)` instead when possible, because `offset`
    /// allows the compiler to optimize better.  If you need to cross object
    /// boundaries, cast the pointer to an integer and do the arithmetic there.
    ///
    /// # Examples
    ///
    /// ```
    /// # let _context = cust::quick_init().unwrap();
    /// use cust::memory::*;
    /// unsafe {
    ///     let mut dev_ptr = cuda_malloc::<u64>(5).unwrap();
    ///     let offset = dev_ptr.wrapping_offset(1); // Points to the 2nd u64 in the buffer
    ///     cuda_free(dev_ptr); // Must free the buffer using the original pointer
    /// }
    /// ```
    pub fn wrapping_offset(self, count: isize) -> Self
    where
        T: Sized,
    {
        unsafe { Self::wrap(self.0.wrapping_offset(count)) }
    }

    /// Calculates the offset from a pointer (convenience for `.offset(count as isize)`).
    ///
    /// `count` is in units of T; e.g. a `count` of 3 represents a pointer
    /// offset of `3 * size_of::<T>()` bytes.
    ///
    /// # Safety
    ///
    /// If any of the following conditions are violated, the result is Undefined
    /// Behavior:
    ///
    /// * Both the starting and resulting pointer must be either in bounds or one
    ///   byte past the end of an allocated object.
    ///
    /// * The computed offset, **in bytes**, cannot overflow an `isize`.
    ///
    /// * The offset being in bounds cannot rely on "wrapping around" the address
    ///   space. That is, the infinite-precision sum must fit in a `usize`.
    ///
    /// Consider using `wrapping_offset` instead if these constraints are
    /// difficult to satisfy. The only advantage of this method is that it
    /// enables more aggressive compiler optimizations.
    ///
    /// # Examples
    ///
    /// ```
    /// # let _context = cust::quick_init().unwrap();
    /// use cust::memory::*;
    /// unsafe {
    ///     let mut dev_ptr = cuda_malloc::<u64>(5).unwrap();
    ///     let offset = dev_ptr.add(1); // Points to the 2nd u64 in the buffer
    ///     cuda_free(dev_ptr); // Must free the buffer using the original pointer
    /// }
    /// ```
    #[allow(clippy::should_implement_trait)]
    pub unsafe fn add(self, count: usize) -> Self
    where
        T: Sized,
    {
        self.offset(count as isize)
    }

    /// Calculates the offset from a pointer (convenience for
    /// `.offset((count as isize).wrapping_neg())`).
    ///
    /// `count` is in units of T; e.g. a `count` of 3 represents a pointer
    /// offset of `3 * size_of::<T>()` bytes.
    ///
    /// # Safety
    ///
    /// If any of the following conditions are violated, the result is Undefined
    /// Behavior:
    ///
    /// * Both the starting and resulting pointer must be either in bounds or one
    ///   byte past the end of an allocated object.
    ///
    /// * The computed offset, **in bytes**, cannot overflow an `isize`.
    ///
    /// * The offset being in bounds cannot rely on "wrapping around" the address
    ///   space. That is, the infinite-precision sum must fit in a `usize`.
    ///
    /// Consider using `wrapping_offset` instead if these constraints are
    /// difficult to satisfy. The only advantage of this method is that it
    /// enables more aggressive compiler optimizations.
    ///
    /// # Examples
    ///
    /// ```
    /// # let _context = cust::quick_init().unwrap();
    /// use cust::memory::*;
    /// unsafe {
    ///     let mut dev_ptr = cuda_malloc::<u64>(5).unwrap();
    ///     let offset = dev_ptr.add(4).sub(3); // Points to the 2nd u64 in the buffer
    ///     cuda_free(dev_ptr); // Must free the buffer using the original pointer
    /// }
    #[allow(clippy::should_implement_trait)]
    pub unsafe fn sub(self, count: usize) -> Self
    where
        T: Sized,
    {
        self.offset((count as isize).wrapping_neg())
    }

    /// Calculates the offset from a pointer using wrapping arithmetic.
    /// (convenience for `.wrapping_offset(count as isize)`)
    ///
    /// `count` is in units of T; e.g. a `count` of 3 represents a pointer
    /// offset of `3 * size_of::<T>()` bytes.
    ///
    /// # Safety
    ///
    /// The resulting pointer does not need to be in bounds, but it is
    /// potentially hazardous to dereference.
    ///
    /// Always use `.add(count)` instead when possible, because `add`
    /// allows the compiler to optimize better.
    ///
    /// # Examples
    ///
    /// ```
    /// # let _context = cust::quick_init().unwrap();
    /// use cust::memory::*;
    /// unsafe {
    ///     let mut dev_ptr = cuda_malloc::<u64>(5).unwrap();
    ///     let offset = dev_ptr.wrapping_add(1); // Points to the 2nd u64 in the buffer
    ///     cuda_free(dev_ptr); // Must free the buffer using the original pointer
    /// }
    /// ```
    pub fn wrapping_add(self, count: usize) -> Self
    where
        T: Sized,
    {
        self.wrapping_offset(count as isize)
    }

    /// Calculates the offset from a pointer using wrapping arithmetic.
    /// (convenience for `.wrapping_offset((count as isize).wrapping_sub())`)
    ///
    /// `count` is in units of T; e.g. a `count` of 3 represents a pointer
    /// offset of `3 * size_of::<T>()` bytes.
    ///
    /// # Safety
    ///
    /// The resulting pointer does not need to be in bounds, but it is
    /// potentially hazardous to dereference (which requires `unsafe`).
    ///
    /// Always use `.sub(count)` instead when possible, because `sub`
    /// allows the compiler to optimize better.
    ///
    /// # Examples
    ///
    /// ```
    /// # let _context = cust::quick_init().unwrap();
    /// use cust::memory::*;
    /// unsafe {
    ///     let mut dev_ptr = cuda_malloc::<u64>(5).unwrap();
    ///     let offset = dev_ptr.wrapping_add(4).wrapping_sub(3); // Points to the 2nd u64 in the buffer
    ///     cuda_free(dev_ptr); // Must free the buffer using the original pointer
    /// }
    /// ```
    pub fn wrapping_sub(self, count: usize) -> Self
    where
        T: Sized,
    {
        self.wrapping_offset((count as isize).wrapping_neg())
    }
}

/// A pointer to unified memory.
///
/// `UnifiedPointer` can be safely dereferenced by the CPU, as the memory allocation it points to is
/// shared between the CPU and the GPU. It can also be safely copied to the device (eg. as part of
/// a kernel launch).
///
/// `UnifiedPointer` is guaranteed to have an equivalent internal representation to a raw pointer.
/// Thus, it can be safely reinterpreted or transmuted to `*mut T`. It is also safe to pass a
/// `UnifiedPointer` through an FFI boundary to C code expecting a `*mut T`. It is
/// thus possible to pass a `UnifiedPointer` to a CUDA kernel written in C.
#[repr(transparent)]
pub struct UnifiedPointer<T: ?Sized>(*mut T);

unsafe impl<T: ?Sized + DeviceCopy> DeviceCopy for UnifiedPointer<T> {}

impl<T: ?Sized> UnifiedPointer<T> {
    /// Wrap the given raw pointer in a UnifiedPointer. The given pointer is assumed to be a valid,
    /// unified-memory pointer or null.
    ///
    /// # Safety
    ///
    /// The given pointer must have been allocated with
    /// [`cuda_malloc_unified`](fn.cuda_malloc_unified.html) or be null.
    ///
    /// # Examples
    ///
    /// ```
    /// # let _context = cust::quick_init().unwrap();
    /// use cust::memory::*;
    /// use std::ptr;
    /// unsafe {
    ///     let null : *mut u64 = ptr::null_mut();
    ///     assert!(UnifiedPointer::wrap(null).is_null());
    /// }
    /// ```
    pub unsafe fn wrap(ptr: *mut T) -> Self {
        UnifiedPointer(ptr)
    }

    /// Returns the contained pointer as a raw pointer.
    ///
    /// # Examples
    ///
    /// ```
    /// # let _context = cust::quick_init().unwrap();
    /// use cust::memory::*;
    /// unsafe {
    ///     let unified_ptr = cuda_malloc_unified::<u64>(1).unwrap();
    ///     let ptr: *const u64 = unified_ptr.as_raw();
    ///     cuda_free_unified(unified_ptr);
    /// }
    /// ```
    pub fn as_raw(self) -> *const T {
        self.0
    }

    /// Returns the contained pointer as a mutable raw pointer.
    ///
    /// # Examples
    ///
    /// ```
    /// # let _context = cust::quick_init().unwrap();
    /// use cust::memory::*;
    /// unsafe {
    ///     let mut unified_ptr = cuda_malloc_unified::<u64>(1).unwrap();
    ///     let ptr: *mut u64 = unified_ptr.as_raw_mut();
    ///     *ptr = 5u64;
    ///     cuda_free_unified(unified_ptr);
    /// }
    /// ```
    pub fn as_raw_mut(&mut self) -> *mut T {
        self.0
    }

    /// Returns true if the pointer is null.
    ///
    /// # Examples
    ///
    /// ```
    /// # let _context = cust::quick_init().unwrap();
    /// use cust::memory::*;
    /// use std::ptr;
    /// unsafe {
    ///     let null : *mut u64 = ptr::null_mut();
    ///     assert!(UnifiedPointer::wrap(null).is_null());
    /// }
    /// ```
    pub fn is_null(self) -> bool {
        self.0.is_null()
    }

    /// Returns a null unified pointer.
    ///
    /// # Examples:
    ///
    /// ```
    /// # let _context = cust::quick_init().unwrap();
    /// use cust::memory::*;
    /// let ptr : UnifiedPointer<u64> = UnifiedPointer::null();
    /// assert!(ptr.is_null());
    /// ```
    pub fn null() -> Self
    where
        T: Sized,
    {
        unsafe { Self::wrap(ptr::null_mut()) }
    }

    /// Calculates the offset from a unified pointer.
    ///
    /// `count` is in units of T; eg. a `count` of 3 represents a pointer offset of
    /// `3 * size_of::<T>()` bytes.
    ///
    /// # Safety
    ///
    /// If any of the following conditions are violated, the result is Undefined
    /// Behavior:
    ///
    /// * Both the starting and resulting pointer must be either in bounds or one
    ///   byte past the end of *the same* allocated object.
    ///
    /// * The computed offset, **in bytes**, cannot overflow an `isize`.
    ///
    /// * The offset being in bounds cannot rely on "wrapping around" the address
    ///   space. That is, the infinite-precision sum, **in bytes** must fit in a usize.
    ///
    /// Consider using `wrapping_offset` instead if these constraints are
    /// difficult to satisfy. The only advantage of this method is that it
    /// enables more aggressive compiler optimizations.
    ///
    /// # Examples
    ///
    /// ```
    /// # let _context = cust::quick_init().unwrap();
    /// use cust::memory::*;
    /// unsafe {
    ///     let mut unified_ptr = cuda_malloc_unified::<u64>(5).unwrap();
    ///     let offset = unified_ptr.offset(1); // Points to the 2nd u64 in the buffer
    ///     cuda_free_unified(unified_ptr); // Must free the buffer using the original pointer
    /// }
    /// ```
    pub unsafe fn offset(self, count: isize) -> Self
    where
        T: Sized,
    {
        Self::wrap(self.0.offset(count))
    }

    /// Calculates the offset from a unified pointer using wrapping arithmetic.
    ///
    /// `count` is in units of T; eg. a `count` of 3 represents a pointer offset of
    /// `3 * size_of::<T>()` bytes.
    ///
    /// # Safety
    ///
    /// The resulting pointer does not need to be in bounds, but it is
    /// potentially hazardous to dereference (which requires `unsafe`).
    /// In particular, the resulting pointer may *not* be used to access a
    /// different allocated object than the one `self` points to. In other
    /// words, `x.wrapping_offset(y.wrapping_offset_from(x))` is
    /// *not* the same as `y`, and dereferencing it is undefined behavior
    /// unless `x` and `y` point into the same allocated object.
    ///
    /// Always use `.offset(count)` instead when possible, because `offset`
    /// allows the compiler to optimize better.  If you need to cross object
    /// boundaries, cast the pointer to an integer and do the arithmetic there.
    ///
    /// # Examples
    ///
    /// ```
    /// # let _context = cust::quick_init().unwrap();
    /// use cust::memory::*;
    /// unsafe {
    ///     let mut unified_ptr = cuda_malloc_unified::<u64>(5).unwrap();
    ///     let offset = unified_ptr.wrapping_offset(1); // Points to the 2nd u64 in the buffer
    ///     cuda_free_unified(unified_ptr); // Must free the buffer using the original pointer
    /// }
    /// ```
    pub fn wrapping_offset(self, count: isize) -> Self
    where
        T: Sized,
    {
        unsafe { Self::wrap(self.0.wrapping_offset(count)) }
    }

    /// Calculates the offset from a pointer (convenience for `.offset(count as isize)`).
    ///
    /// `count` is in units of T; e.g. a `count` of 3 represents a pointer
    /// offset of `3 * size_of::<T>()` bytes.
    ///
    /// # Safety
    ///
    /// If any of the following conditions are violated, the result is Undefined
    /// Behavior:
    ///
    /// * Both the starting and resulting pointer must be either in bounds or one
    ///   byte past the end of an allocated object.
    ///
    /// * The computed offset, **in bytes**, cannot overflow an `isize`.
    ///
    /// * The offset being in bounds cannot rely on "wrapping around" the address
    ///   space. That is, the infinite-precision sum must fit in a `usize`.
    ///
    /// Consider using `wrapping_offset` instead if these constraints are
    /// difficult to satisfy. The only advantage of this method is that it
    /// enables more aggressive compiler optimizations.
    ///
    /// # Examples
    ///
    /// ```
    /// # let _context = cust::quick_init().unwrap();
    /// use cust::memory::*;
    /// unsafe {
    ///     let mut unified_ptr = cuda_malloc_unified::<u64>(5).unwrap();
    ///     let offset = unified_ptr.add(1); // Points to the 2nd u64 in the buffer
    ///     cuda_free_unified(unified_ptr); // Must free the buffer using the original pointer
    /// }
    /// ```
    #[allow(clippy::should_implement_trait)]
    pub unsafe fn add(self, count: usize) -> Self
    where
        T: Sized,
    {
        self.offset(count as isize)
    }

    /// Calculates the offset from a pointer (convenience for
    /// `.offset((count as isize).wrapping_neg())`).
    ///
    /// `count` is in units of T; e.g. a `count` of 3 represents a pointer
    /// offset of `3 * size_of::<T>()` bytes.
    ///
    /// # Safety
    ///
    /// If any of the following conditions are violated, the result is Undefined
    /// Behavior:
    ///
    /// * Both the starting and resulting pointer must be either in bounds or one
    ///   byte past the end of an allocated object.
    ///
    /// * The computed offset, **in bytes**, cannot overflow an `isize`.
    ///
    /// * The offset being in bounds cannot rely on "wrapping around" the address
    ///   space. That is, the infinite-precision sum must fit in a `usize`.
    ///
    /// Consider using `wrapping_offset` instead if these constraints are
    /// difficult to satisfy. The only advantage of this method is that it
    /// enables more aggressive compiler optimizations.
    ///
    /// # Examples
    ///
    /// ```
    /// # let _context = cust::quick_init().unwrap();
    /// use cust::memory::*;
    /// unsafe {
    ///     let mut unified_ptr = cuda_malloc_unified::<u64>(5).unwrap();
    ///     let offset = unified_ptr.add(4).sub(3); // Points to the 2nd u64 in the buffer
    ///     cuda_free_unified(unified_ptr); // Must free the buffer using the original pointer
    /// }
    #[allow(clippy::should_implement_trait)]
    pub unsafe fn sub(self, count: usize) -> Self
    where
        T: Sized,
    {
        self.offset((count as isize).wrapping_neg())
    }

    /// Calculates the offset from a pointer using wrapping arithmetic.
    /// (convenience for `.wrapping_offset(count as isize)`)
    ///
    /// `count` is in units of T; e.g. a `count` of 3 represents a pointer
    /// offset of `3 * size_of::<T>()` bytes.
    ///
    /// # Safety
    ///
    /// The resulting pointer does not need to be in bounds, but it is
    /// potentially hazardous to dereference.
    ///
    /// Always use `.add(count)` instead when possible, because `add`
    /// allows the compiler to optimize better.
    ///
    /// # Examples
    ///
    /// ```
    /// # let _context = cust::quick_init().unwrap();
    /// use cust::memory::*;
    /// unsafe {
    ///     let mut unified_ptr = cuda_malloc_unified::<u64>(5).unwrap();
    ///     let offset = unified_ptr.wrapping_add(1); // Points to the 2nd u64 in the buffer
    ///     cuda_free_unified(unified_ptr); // Must free the buffer using the original pointer
    /// }
    /// ```
    pub fn wrapping_add(self, count: usize) -> Self
    where
        T: Sized,
    {
        self.wrapping_offset(count as isize)
    }

    /// Calculates the offset from a pointer using wrapping arithmetic.
    /// (convenience for `.wrapping_offset((count as isize).wrapping_sub())`)
    ///
    /// `count` is in units of T; e.g. a `count` of 3 represents a pointer
    /// offset of `3 * size_of::<T>()` bytes.
    ///
    /// # Safety
    ///
    /// The resulting pointer does not need to be in bounds, but it is
    /// potentially hazardous to dereference (which requires `unsafe`).
    ///
    /// Always use `.sub(count)` instead when possible, because `sub`
    /// allows the compiler to optimize better.
    ///
    /// # Examples
    ///
    /// ```
    /// # let _context = cust::quick_init().unwrap();
    /// use cust::memory::*;
    /// unsafe {
    ///     let mut unified_ptr = cuda_malloc_unified::<u64>(5).unwrap();
    ///     let offset = unified_ptr.wrapping_add(4).wrapping_sub(3); // Points to the 2nd u64 in the buffer
    ///     cuda_free_unified(unified_ptr); // Must free the buffer using the original pointer
    /// }
    /// ```
    pub fn wrapping_sub(self, count: usize) -> Self
    where
        T: Sized,
    {
        self.wrapping_offset((count as isize).wrapping_neg())
    }
}
