//! Access to CUDA's memory allocation and transfer functions.
//!
//! The memory module provides a safe wrapper around CUDA's memory allocation and transfer functions.
//! This includes access to device memory, unified memory, and page-locked host memory.
//!
//! # Device Memory
//!
//! Device memory is just what it sounds like - memory allocated on the device. Device memory
//! cannot be accessed from the host directly, but data can be copied to and from the device.
//! cust exposes device memory through the [`DBox`](struct.DBox.html) and
//! [`DeviceBuffer`](struct.DeviceBuffer.html) structures. Pointers to device memory are
//! represented by [`DevicePointer`](struct.DevicePointer.html), while slices in device memory are
//! represented by [`DeviceSlice`](struct.DeviceSlice.html).
//!
//! # Unified Memory
//!
//! Unified memory is a memory allocation which can be read from and written to by both the host
//! and the device. When the host (or device) attempts to access a page of unified memory, it is
//! seamlessly transferred from host RAM to device RAM or vice versa. The programmer may also
//! choose to explicitly prefetch data to one side or another (though this is not currently exposed
//! through cust). cust exposes unified memory through the
//! [`UnifiedBox`](struct.UnifiedBox.html) and [`UnifiedBuffer`](struct.UnifiedBuffer.html)
//! structures, and pointers to unified memory are represented by
//! [`UnifiedPointer`](struct.UnifiedPointer.html). Since unified memory is accessible to the host,
//! slices in unified memory are represented by normal Rust slices.
//!
//! Unified memory is generally easier to use than device memory, but there are drawbacks. It is
//! possible to allocate more memory than is available on the card, and this can result in very slow
//! paging behavior. Additionally, it can require careful use of prefetching to achieve optimum
//! performance. Finally, unified memory is not supported on some older systems.
//!
//! ## Warning
//!
//! ⚠️ **On certain systems/OSes/GPUs, accessing Unified memory from the CPU while the GPU is currently
//! using it (e.g. before stream synchronization) will cause a Page Error/Segfault. For this reason,
//! we strongly suggest to treat unified memory as exclusive to the GPU while it is being used by a kernel** ⚠️
//!
//! This is not considered Undefined Behavior because the behavior is always "either works, or yields a page error/segfault",
//! doing this will never corrupt memory or cause other undesireable behavior.
//!
//! # Page-locked Host Memory
//!
//! Page-locked memory is memory that the operating system has locked into physical RAM, and will
//! not page out to disk. When copying data from the process' memory space to the device, the CUDA
//! driver needs to first copy the data to a page-locked region of host memory, then initiate a DMA
//! transfer to copy the data to the device itself. Likewise, when transferring from device to host,
//! the driver copies the data into page-locked host memory then into the normal memory space. This
//! extra copy can be eliminated if the data is loaded or generated directly into page-locked
//! memory. cust exposes page-locked memory through the
//! [`LockedBuffer`](struct.LockedBuffer.html) struct.
//!
//! For example, if the programmer needs to read an array of bytes from disk and transfer it to the
//! device, it would be best to create a `LockedBuffer`, load the bytes directly into the
//! `LockedBuffer`, and then copy them to a `DeviceBuffer`. If the bytes are in a `Vec<u8>`, there
//! would be no advantage to using a `LockedBuffer`.
//!
//! However, since the OS cannot page out page-locked memory, excessive use can slow down the entire
//! system (including other processes) as physical RAM is tied up.  Therefore, page-locked memory
//! should be used sparingly.
//!
//! # FFI Information
//!
//! The internal representations of `DevicePointer<T>` and `UnifiedPointer<T>` are guaranteed to be
//! the same as `*mut T` and they can be safely passed through an FFI boundary to code expecting
//! raw pointers (though keep in mind that device-only pointers cannot be dereferenced on the CPU).
//! This is important when launching kernels written in C.
//!
//! As with regular Rust, all other types (eg. `DeviceBuffer` or `UnifiedBox`) are not FFI-safe.
//! Their internal representations are not guaranteed to be anything in particular, and are not
//! guaranteed to be the same in different versions of cust. If you need to pass them through
//! an FFI boundary, you must convert them to FFI-safe primitives yourself. For example, with
//! `UnifiedBuffer`, use the `as_unified_ptr()` and `len()` functions to get the primitives, and
//! `mem::forget()` the Buffer so that it isn't dropped. Again, as with regular Rust, the caller is
//! responsible for reconstructing the `UnifiedBuffer` using `from_raw_parts()` and dropping it to
//! ensure that the memory allocation is safely cleaned up.

pub mod array;

mod device;
mod locked;
mod malloc;
mod pointer;
mod unified;

pub use self::device::*;
pub use self::locked::*;
pub use self::malloc::*;
pub use self::pointer::*;
pub use self::unified::*;

use core::marker::PhantomData;
use core::num::*;

/// A trait describing a generic buffer that can be accessed from the GPU. This could be either a [`UnifiedBuffer`]
/// or a regular [`DBuffer`].
pub trait GpuBuffer<T: DeviceCopy>: private::Sealed {
    /// Obtains a device pointer to this buffer, not requiring `&mut self`. This means
    /// the buffer must be used only for immutable reads, and must not be written to.
    unsafe fn as_device_ptr(&self) -> DevicePointer<T>;
    fn as_device_ptr_mut(&mut self) -> DevicePointer<T>;
    fn len(&self) -> usize;
}

impl<T: DeviceCopy> GpuBuffer<T> for DBuffer<T> {
    unsafe fn as_device_ptr(&self) -> DevicePointer<T> {
        DevicePointer::wrap((**self).as_ptr() as *mut _)
    }

    fn as_device_ptr_mut(&mut self) -> DevicePointer<T> {
        (**self).as_device_ptr()
    }

    fn len(&self) -> usize {
        (**self).len()
    }
}

impl<T: DeviceCopy> GpuBuffer<T> for UnifiedBuffer<T> {
    unsafe fn as_device_ptr(&self) -> DevicePointer<T> {
        DevicePointer::wrap(self.as_ptr() as *mut _)
    }

    fn as_device_ptr_mut(&mut self) -> DevicePointer<T> {
        // SAFETY: unified pointers can be dereferenced from the gpu.
        unsafe { DevicePointer::wrap(self.as_ptr() as *mut _) }
    }

    fn len(&self) -> usize {
        (**self).len()
    }
}

/// A trait describing a generic pointer that can be accessed from the GPU. This could be either a [`UnifiedBox`]
/// or a regular [`DBox`].
pub trait GpuBox<T: DeviceCopy>: private::Sealed {
    /// Obtains a device pointer to this value, not requiring `&mut self`. This means
    /// the value must be used only for immutable reads, and must not be written to.
    unsafe fn as_device_ptr(&self) -> DevicePointer<T>;
    fn as_device_ptr_mut(&mut self) -> DevicePointer<T>;
}

impl<T: DeviceCopy> GpuBox<T> for DBox<T> {
    unsafe fn as_device_ptr(&self) -> DevicePointer<T> {
        self.ptr
    }

    fn as_device_ptr_mut(&mut self) -> DevicePointer<T> {
        DBox::as_device_ptr(self)
    }
}

impl<T: DeviceCopy> GpuBox<T> for UnifiedBox<T> {
    unsafe fn as_device_ptr(&self) -> DevicePointer<T> {
        DevicePointer::wrap(self.ptr.as_raw() as *mut _)
    }

    fn as_device_ptr_mut(&mut self) -> DevicePointer<T> {
        // SAFETY: unified pointers can be dereferenced from the gpu.
        unsafe { DevicePointer::wrap(self.ptr.as_raw() as *mut _) }
    }
}

mod private {
    use super::{DBox, DBuffer, DeviceCopy, UnifiedBox, UnifiedBuffer};

    pub trait Sealed {}
    impl<T: DeviceCopy> Sealed for UnifiedBuffer<T> {}
    impl<T: DeviceCopy> Sealed for DBuffer<T> {}
    impl<T: DeviceCopy> Sealed for UnifiedBox<T> {}
    impl<T: DeviceCopy> Sealed for DBox<T> {}
}

/// Marker trait for types which can safely be copied to or from a CUDA device.
///
/// A type can be safely copied if its value can be duplicated simply by copying bits and if it does
/// not contain a reference to memory which is not accessible to the device. Additionally, the
/// DeviceCopy trait does not imply copy semantics as the Copy trait does.
///
/// ## How can I implement DeviceCopy?
///
/// There are two ways to implement DeviceCopy on your type. The simplest is to use `derive`:
///
/// ```
/// use cust::DeviceCopy;
///
/// #[derive(Clone, DeviceCopy)]
/// struct MyStruct(u64);
///
/// # fn main () {}
/// ```
///
/// This is safe because the `DeviceCopy` derive macro will check that all fields of the struct,
/// enum or union implement `DeviceCopy`. For example, this fails to compile, because `Vec` cannot
/// be copied to the device:
///
/// ```compile_fail
/// use cust::DeviceCopy;
///
/// #[derive(Clone, DeviceCopy)]
/// struct MyStruct(Vec<u64>);
/// # fn main () {}
/// ```
///
/// You can also implement `DeviceCopy` unsafely:
///
/// ```
/// use cust::memory::DeviceCopy;
///
/// #[derive(Clone)]
/// struct MyStruct(u64);
///
/// unsafe impl DeviceCopy for MyStruct { }
/// # fn main () {}
/// ```
///
/// ## What is the difference between `DeviceCopy` and `Copy`?
///
/// `DeviceCopy` is stricter than `Copy`. `DeviceCopy` must only be implemented for types which
/// do not contain references or raw pointers to non-device-accessible memory. `DeviceCopy` also
/// does not imply copy semantics - that is, `DeviceCopy` values are not implicitly copied on
/// assignment the way that `Copy` values are. This is helpful, as it may be desirable to implement
/// `DeviceCopy` for large structures that would be inefficient to copy for every assignment.
///
/// ## When can't my type be `DeviceCopy`?
///
/// Some types cannot be safely copied to the device. For example, copying `&T` would create an
/// invalid reference on the device which would segfault if dereferenced. Generalizing this, any
/// type implementing `Drop` cannot be `DeviceCopy` since it is responsible for some resource that
/// would not be available on the device.
pub unsafe trait DeviceCopy: Copy {}

macro_rules! impl_device_copy {
    ($($t:ty)*) => {
        $(
            unsafe impl DeviceCopy for $t {}
        )*
    }
}

impl_device_copy!(
    usize u8 u16 u32 u64 u128
    isize i8 i16 i32 i64 i128
    f32 f64
    bool char

    NonZeroU8 NonZeroU16 NonZeroU32 NonZeroU64 NonZeroU128
);
unsafe impl<T: DeviceCopy> DeviceCopy for Option<T> {}
unsafe impl<L: DeviceCopy, R: DeviceCopy> DeviceCopy for Result<L, R> {}
unsafe impl<T: ?Sized + DeviceCopy> DeviceCopy for PhantomData<T> {}
unsafe impl<T: DeviceCopy> DeviceCopy for Wrapping<T> {}
unsafe impl<T: DeviceCopy, const N: usize> DeviceCopy for [T; N] {}
unsafe impl DeviceCopy for () {}
unsafe impl<A: DeviceCopy, B: DeviceCopy> DeviceCopy for (A, B) {}
unsafe impl<A: DeviceCopy, B: DeviceCopy, C: DeviceCopy> DeviceCopy for (A, B, C) {}
unsafe impl<A: DeviceCopy, B: DeviceCopy, C: DeviceCopy, D: DeviceCopy> DeviceCopy
    for (A, B, C, D)
{
}
unsafe impl<A: DeviceCopy, B: DeviceCopy, C: DeviceCopy, D: DeviceCopy, E: DeviceCopy> DeviceCopy
    for (A, B, C, D, E)
{
}
unsafe impl<A: DeviceCopy, B: DeviceCopy, C: DeviceCopy, D: DeviceCopy, E: DeviceCopy, F: DeviceCopy>
    DeviceCopy for (A, B, C, D, E, F)
{
}
unsafe impl<
        A: DeviceCopy,
        B: DeviceCopy,
        C: DeviceCopy,
        D: DeviceCopy,
        E: DeviceCopy,
        F: DeviceCopy,
        G: DeviceCopy,
    > DeviceCopy for (A, B, C, D, E, F, G)
{
}
unsafe impl<
        A: DeviceCopy,
        B: DeviceCopy,
        C: DeviceCopy,
        D: DeviceCopy,
        E: DeviceCopy,
        F: DeviceCopy,
        G: DeviceCopy,
        H: DeviceCopy,
    > DeviceCopy for (A, B, C, D, E, F, G, H)
{
}

#[cfg(feature = "vek")]
macro_rules! impl_device_copy_vek {
    ($($strukt:ident),* $(,)?) => {
        $(
            unsafe impl<T: DeviceCopy> DeviceCopy for $strukt<T> {}
        )*
    }
}

#[cfg(feature = "vek")]
use vek::*;

#[cfg(feature = "vek")]
impl_device_copy_vek! {
    Vec2, Vec3, Vec4, Extent2, Extent3, Rgb, Rgba,
    Mat2, Mat3, Mat4,
    CubicBezier2, CubicBezier3,
    Quaternion,
}
