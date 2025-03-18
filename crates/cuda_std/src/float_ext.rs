//! Extension trait for [`f32`] and [`f64`], providing high level wrappers on top of
//! raw libdevice intrinsics from [`intrinsics`](crate::intrinsics).

use cuda_std_macros::gpu_only;

#[cfg(target_arch = "nvptx64")]
use crate::intrinsics as raw;

// allows us to add new functions to the trait at any time without needing a new major version.
mod private {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
}

/// Extension trait for [`f32`] and [`f64`] which provides high level functions for
/// low level intrinsics for common math operations. You should generally use
/// these functions over "manual" implementations because they are often much faster.
///
/// Note that these link to libdevice intrinsics, so these functions cannot be used in cpu code,
/// or they will fail to link.
pub trait FloatExt: Sized + private::Sealed {
    /// The cosine of `self * pi` (measured in radians).
    fn cospi(self) -> Self;
    /// The value of the error function `2/\sqrt{pi} \int_0^x e^{-t^2} dt`.
    fn error_function(self) -> Self;
    /// The value of the complementary error function `1 - [error_function](Self::error_function)`.
    fn complementary_error_function(self) -> Self;
    /// Tries to find the value of `x` that satisfies `Self = complementary_error_function(x)`. Where
    /// `Self` is in the interval [`0`, `2`].
    fn inv_complementary_error_function(self) -> Self;
    /// The value of the scaled complementary error function, `e^x^2 * self.complementary_error_function()`.
    fn scaled_complementary_error_function(self) -> Self;
    /// Decomposes self into a fractional component (which will be either 0, or in the range of 0.5..1.0) and an exponent.
    /// Aka, `self = fractional * 2^exponent`.
    fn frexp(self) -> (Self, i32);
    /// The unbiased integer exponent of self.
    fn unbiased_exp(self) -> i32;
    /// The value of the bessel function of the first kind of order 0 for self. `J_0(self)`.
    fn j0(self) -> Self;
    /// The value of the bessel function of the first kind of order 1 for self. `J_1(self)`.
    fn j1(self) -> Self;
    /// The value of the bessel function of the first kind of order n for self. `J_n(self)`.
    fn jn(self, order: i32) -> Self;
    /// The value of `self * 2^exp`.
    fn ldexp(self, exp: i32) -> Self;
    /// The natural logarithm of the absolute value of the gamma function. `log_e (\int_0^\inf e^-t t^{x-1} dt)`
    fn log_gamma(self) -> Self;
    /// The natural logarithm of `1 + self`, `log_e(1 + self)`
    fn log1p(self) -> Self;
    /// The cumulative distribution function of the standard normal distribution for self. `\phi(self)`.
    fn norm_cdf(self) -> Self;
    /// The inverse cumulative distribution function of the standard normal distribution for self. `\phi^-1(self)`.
    /// This function is defined for input values in the interval (0, 1).
    fn inv_norm_cdf(self) -> Self;
    /// The reciprocal cube root of self.
    fn rcbrt(self) -> Self;
    /// Clamp self to [+0.0, 1.0].
    fn saturate(self) -> Self;
    /// Scales self by `2^n` (`self * 2^n`) efficiently.
    fn scale_by_n(self, exp: i32) -> Self;
    /// The sine and cosine of `self * pi` (measured in radians).
    fn sincospi(self) -> (Self, Self);
    /// The sine of `self * pi` (measured in radians).
    fn sinpi(self) -> Self;
    /// The gamma function of self, `\int_0^\inf e^-t t^{x-1} dt`.
    fn gamma(self) -> Self;
    /// The value of the bessel function of the second kind of order 0 for self. `Y_0(self)`.
    fn y0(self) -> Self;
    /// The value of the bessel function of the second kind of order 1 for self. `Y_1(self)`.
    fn y1(self) -> Self;
    /// The value of the bessel function of the second kind of order n for self. `Y_n(self)`.
    fn yn(self, order: i32) -> Self;
}

impl FloatExt for f64 {
    #[gpu_only]
    fn cospi(self) -> Self {
        unsafe { raw::cospi(self) }
    }

    #[gpu_only]
    fn error_function(self) -> Self {
        unsafe { raw::erf(self) }
    }

    #[gpu_only]
    fn complementary_error_function(self) -> Self {
        unsafe { raw::erfc(self) }
    }

    #[gpu_only]
    fn inv_complementary_error_function(self) -> Self {
        unsafe { raw::erfcinv(self) }
    }

    #[gpu_only]
    fn scaled_complementary_error_function(self) -> Self {
        unsafe { raw::erfcx(self) }
    }

    #[gpu_only]
    fn frexp(self) -> (Self, i32) {
        let mut exp = 0;
        unsafe {
            let fract = raw::frexp(self, &mut exp as *mut i32);
            (fract, exp)
        }
    }

    #[gpu_only]
    fn unbiased_exp(self) -> i32 {
        unsafe { raw::ilogb(self) }
    }

    #[gpu_only]
    fn j0(self) -> Self {
        unsafe { raw::j0(self) }
    }

    #[gpu_only]
    fn j1(self) -> Self {
        unsafe { raw::j1(self) }
    }

    #[gpu_only]
    fn jn(self, order: i32) -> Self {
        unsafe { raw::jn(order, self) }
    }

    #[gpu_only]
    fn ldexp(self, exp: i32) -> Self {
        unsafe { raw::ldexp(self, exp) }
    }

    #[gpu_only]
    fn log_gamma(self) -> Self {
        unsafe { raw::lgamma(self) }
    }

    #[gpu_only]
    fn log1p(self) -> Self {
        unsafe { raw::log1p(self) }
    }

    #[gpu_only]
    fn norm_cdf(self) -> Self {
        unsafe { raw::normcdf(self) }
    }

    #[gpu_only]
    fn inv_norm_cdf(self) -> Self {
        unsafe { raw::normcdfinv(self) }
    }

    #[gpu_only]
    fn rcbrt(self) -> Self {
        unsafe { raw::rcbrt(self) }
    }

    #[gpu_only]
    fn saturate(self) -> Self {
        // this intrinsic doesnt actually exit on f64, so implement it as clamp on f64
        self.clamp(0.0, 1.0)
    }

    #[gpu_only]
    fn scale_by_n(self, exp: i32) -> Self {
        unsafe { raw::scalbn(self, exp) }
    }

    #[gpu_only]
    fn sincospi(self) -> (Self, Self) {
        let mut sin = 0.0;
        let mut cos = 0.0;
        unsafe {
            raw::sincospi(self, &mut sin as *mut f64, &mut cos as *mut f64);
        }
        (sin, cos)
    }

    #[gpu_only]
    fn sinpi(self) -> Self {
        unsafe { raw::sinpi(self) }
    }

    #[gpu_only]
    fn gamma(self) -> Self {
        unsafe { raw::tgamma(self) }
    }

    #[gpu_only]
    fn y0(self) -> Self {
        unsafe { raw::y0(self) }
    }

    #[gpu_only]
    fn y1(self) -> Self {
        unsafe { raw::y1(self) }
    }

    #[gpu_only]
    fn yn(self, order: i32) -> Self {
        unsafe { raw::yn(order, self) }
    }
}

impl FloatExt for f32 {
    #[gpu_only]
    fn cospi(self) -> Self {
        unsafe { raw::cospif(self) }
    }

    #[gpu_only]
    fn error_function(self) -> Self {
        unsafe { raw::erff(self) }
    }

    #[gpu_only]
    fn complementary_error_function(self) -> Self {
        unsafe { raw::erfcf(self) }
    }

    #[gpu_only]
    fn inv_complementary_error_function(self) -> Self {
        unsafe { raw::erfcinvf(self) }
    }

    #[gpu_only]
    fn scaled_complementary_error_function(self) -> Self {
        unsafe { raw::erfcxf(self) }
    }

    #[gpu_only]
    fn frexp(self) -> (Self, i32) {
        let mut exp = 0;
        unsafe {
            let fract = raw::frexpf(self, &mut exp as *mut i32);
            (fract, exp)
        }
    }

    #[gpu_only]
    fn unbiased_exp(self) -> i32 {
        unsafe { raw::ilogbf(self) }
    }

    #[gpu_only]
    fn j0(self) -> Self {
        unsafe { raw::j0f(self) }
    }

    #[gpu_only]
    fn j1(self) -> Self {
        unsafe { raw::j1f(self) }
    }

    #[gpu_only]
    fn jn(self, order: i32) -> Self {
        unsafe { raw::jnf(order, self) }
    }

    #[gpu_only]
    fn ldexp(self, exp: i32) -> Self {
        unsafe { raw::ldexpf(self, exp) }
    }

    #[gpu_only]
    fn log_gamma(self) -> Self {
        unsafe { raw::lgammaf(self) }
    }

    #[gpu_only]
    fn log1p(self) -> Self {
        unsafe { raw::log1pf(self) }
    }

    #[gpu_only]
    fn norm_cdf(self) -> Self {
        unsafe { raw::normcdff(self) }
    }

    #[gpu_only]
    fn inv_norm_cdf(self) -> Self {
        unsafe { raw::normcdfinvf(self) }
    }

    #[gpu_only]
    fn rcbrt(self) -> Self {
        unsafe { raw::rcbrtf(self) }
    }

    #[gpu_only]
    fn saturate(self) -> Self {
        unsafe { raw::saturatef(self) }
    }

    #[gpu_only]
    fn scale_by_n(self, exp: i32) -> Self {
        unsafe { raw::scalbnf(self, exp) }
    }

    #[gpu_only]
    fn sincospi(self) -> (Self, Self) {
        let mut sin = 0.0;
        let mut cos = 0.0;
        unsafe {
            raw::sincospif(self, &mut sin as *mut f32, &mut cos as *mut f32);
        }
        (sin, cos)
    }

    #[gpu_only]
    fn sinpi(self) -> Self {
        unsafe { raw::sinpif(self) }
    }

    #[gpu_only]
    fn gamma(self) -> Self {
        unsafe { raw::tgammaf(self) }
    }

    #[gpu_only]
    fn y0(self) -> Self {
        unsafe { raw::y0f(self) }
    }

    #[gpu_only]
    fn y1(self) -> Self {
        unsafe { raw::y1f(self) }
    }

    #[gpu_only]
    fn yn(self, order: i32) -> Self {
        unsafe { raw::ynf(order, self) }
    }
}
