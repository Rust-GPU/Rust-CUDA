#![no_std]

pub use _hidden::*;
pub use cust_derive::DeviceCopyCore as DeviceCopy;

#[doc(hidden)]
pub mod _hidden {
    use core::marker::PhantomData;
    use core::mem::MaybeUninit;
    use core::num::*;

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
    ///
    /// # Safety
    ///
    /// The type being implemented must hold no references to CPU data.
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
    unsafe impl<T> DeviceCopy for *const T {}
    unsafe impl<T> DeviceCopy for *mut T {}
    unsafe impl<T: DeviceCopy> DeviceCopy for MaybeUninit<T> {}
    unsafe impl<T: DeviceCopy> DeviceCopy for Option<T> {}
    unsafe impl<L: DeviceCopy, R: DeviceCopy> DeviceCopy for Result<L, R> {}
    unsafe impl<T: ?Sized + DeviceCopy> DeviceCopy for PhantomData<T> {}
    // Allow DeviceCopy for lifetime constraint markers
    unsafe impl DeviceCopy for PhantomData<&()> {}
    unsafe impl<T: DeviceCopy> DeviceCopy for Wrapping<T> {}
    unsafe impl<T: DeviceCopy, const N: usize> DeviceCopy for [T; N] {}
    unsafe impl DeviceCopy for () {}
    unsafe impl<A: DeviceCopy, B: DeviceCopy> DeviceCopy for (A, B) {}
    unsafe impl<A: DeviceCopy, B: DeviceCopy, C: DeviceCopy> DeviceCopy for (A, B, C) {}
    unsafe impl<A: DeviceCopy, B: DeviceCopy, C: DeviceCopy, D: DeviceCopy> DeviceCopy
        for (A, B, C, D)
    {
    }
    unsafe impl<A: DeviceCopy, B: DeviceCopy, C: DeviceCopy, D: DeviceCopy, E: DeviceCopy>
        DeviceCopy for (A, B, C, D, E)
    {
    }
    unsafe impl<
            A: DeviceCopy,
            B: DeviceCopy,
            C: DeviceCopy,
            D: DeviceCopy,
            E: DeviceCopy,
            F: DeviceCopy,
        > DeviceCopy for (A, B, C, D, E, F)
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

    macro_rules! impl_device_copy_generic {
    ($($($strukt:ident)::+),* $(,)?) => {
        $(
            unsafe impl<T: DeviceCopy> DeviceCopy for $($strukt)::+<T> {}
        )*
    }
}

    macro_rules! impl_device_copy {
    ($($strukt:ty),* $(,)?) => {
        $(
            unsafe impl DeviceCopy for $strukt {}
        )*
    }
}

    #[cfg(feature = "vek")]
    use vek::*;

    #[cfg(feature = "vek")]
    impl_device_copy_generic! {
        Vec2, Vec3, Vec4, Extent2, Extent3,
        Mat2, Mat3, Mat4,
        CubicBezier2, CubicBezier3,
        Quaternion,
    }

    #[cfg(feature = "glam")]
    impl_device_copy! {
        glam::Vec2, glam::Vec3, glam::Vec4, glam::IVec2, glam::IVec3, glam::IVec4, glam::Mat3, glam::Mat4
    }

    #[cfg(feature = "mint")]
    impl_device_copy_generic! {
        mint::Vector2, mint::Vector3, mint::Vector4,
        mint::ColumnMatrix2, mint::ColumnMatrix3, mint::ColumnMatrix4, mint::ColumnMatrix3x4,
        mint::RowMatrix2, mint::RowMatrix3, mint::RowMatrix4, mint::RowMatrix3x4,
    }

    #[cfg(feature = "half")]
    unsafe impl DeviceCopy for half::f16 {}
    #[cfg(feature = "half")]
    unsafe impl DeviceCopy for half::bf16 {}

    #[cfg(feature = "num-complex")]
    impl_device_copy_generic! {
        num_complex::Complex
    }
}
