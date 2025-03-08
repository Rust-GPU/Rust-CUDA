//! Trait for float intrinsics for making floats work in no_std gpu environments.
//!
//! Float functions are mapped directly to libdevice intrinsics on nvptx and
//! their std counterparts on non-nvptx.

/// std float intrinsics implemented using libdevice intrinsics so they can be used
/// from GPU no_std crates. Falls back to stdlib implementation on non-nvptx.
pub trait GpuFloat: Copy + PartialOrd + private::Sealed {
    fn floor(self) -> Self;
    fn ceil(self) -> Self;
    fn round(self) -> Self;
    fn trunc(self) -> Self;
    fn fract(self) -> Self;
    fn abs(self) -> Self;
    fn signum(self) -> Self;
    fn copysign(self, sign: Self) -> Self;
    fn mul_add(self, a: Self, b: Self) -> Self;
    fn div_euclid(self, rhs: Self) -> Self;
    fn rem_euclid(self, rhs: Self) -> Self;
    fn powi(self, n: i32) -> Self;
    fn powf(self, n: Self) -> Self;
    fn sqrt(self) -> Self;
    fn exp(self) -> Self;
    fn exp2(self) -> Self;
    fn ln(self) -> Self;
    fn log(self, base: Self) -> Self;
    fn log2(self) -> Self;
    fn log10(self) -> Self;
    fn cbrt(self) -> Self;
    fn hypot(self, other: Self) -> Self;
    fn sin(self) -> Self;
    fn cos(self) -> Self;
    fn tan(self) -> Self;
    fn asin(self) -> Self;
    fn acos(self) -> Self;
    fn atan(self) -> Self;
    fn atan2(self, other: Self) -> Self;
    fn sin_cos(self) -> (Self, Self);
    fn exp_m1(self) -> Self;
    fn ln_1p(self) -> Self;
    fn sinh(self) -> Self;
    fn cosh(self) -> Self;
    fn tanh(self) -> Self;
    fn asinh(self) -> Self;
    fn acosh(self) -> Self;
    fn atanh(self) -> Self;
}

mod private {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
}

macro_rules! f32_intrinsic {
    ($self:expr, $func:ident($($param:expr),*)) => {{
        #[cfg(not(target_arch = "nvptx64"))]
        let val = $self.$func($($param),*);
        #[cfg(target_arch = "nvptx64")]
        let val = paste::paste! { unsafe { intrinsics::[<$func f>]($self, $($param),*)} };
        val
    }};
}

macro_rules! f64_intrinsic {
    ($self:expr, $func:ident($($param:expr),*)) => {{
        #[cfg(not(target_arch = "nvptx64"))]
        let val = $self.$func($($param),*);
        #[cfg(target_arch = "nvptx64")]
        let val = unsafe { intrinsics::$func($self, $($param),*)};
        val
    }};
}

#[cfg(target_arch = "nvptx64")]
use crate::intrinsics;

impl GpuFloat for f32 {
    /// Returns the largest integer less than or equal to a number.
    #[must_use = "method returns a new number and does not mutate the original value"]
    #[inline]
    fn floor(self) -> f32 {
        f32_intrinsic!(self, floor())
    }

    /// Returns the smallest integer greater than or equal to a number.
    #[must_use = "method returns a new number and does not mutate the original value"]
    #[inline]
    fn ceil(self) -> f32 {
        f32_intrinsic!(self, ceil())
    }

    /// Returns the nearest integer to a number. Round half-way cases away from
    /// `0.0`.
    #[must_use = "method returns a new number and does not mutate the original value"]
    #[inline]
    fn round(self) -> f32 {
        f32_intrinsic!(self, round())
    }

    /// Returns the integer part of a number.
    #[must_use = "method returns a new number and does not mutate the original value"]
    #[inline]
    fn trunc(self) -> f32 {
        f32_intrinsic!(self, trunc())
    }

    /// Returns the fractional part of a number.
    #[must_use = "method returns a new number and does not mutate the original value"]
    #[inline]
    fn fract(self) -> f32 {
        self - self.trunc()
    }

    /// Computes the absolute value of `self`. Returns `NAN` if the
    /// number is `NAN`.
    #[must_use = "method returns a new number and does not mutate the original value"]
    #[inline]
    fn abs(self) -> f32 {
        #[cfg(not(target_arch = "nvptx64"))]
        let val = self.abs();
        #[cfg(target_arch = "nvptx64")]
        let val = { unsafe { intrinsics::fabsf(self) } };
        val
    }

    /// Returns a number that represents the sign of `self`.
    ///
    /// - `1.0` if the number is positive, `+0.0` or `INFINITY`
    /// - `-1.0` if the number is negative, `-0.0` or `NEG_INFINITY`
    /// - `NAN` if the number is `NAN`intrinsics
    #[must_use = "method returns a new number and does not mutate the original value"]
    #[inline]
    fn signum(self) -> f32 {
        if self.is_nan() {
            Self::NAN
        } else {
            1.0_f32.copysign(self)
        }
    }

    /// Returns a number composed of the magnitude of `self` and the sign of
    /// `sign`.
    ///
    /// Equal to `self` if the sign of `self` and `sign` are the same, otherwise
    /// equal to `-self`. If `self` is a `NAN`, then a `NAN` with the sign of
    /// `sign` is returned.
    #[must_use = "method returns a new number and does not mutate the original value"]
    #[inline]
    fn copysign(self, sign: f32) -> f32 {
        f32_intrinsic!(self, copysign(sign))
    }

    /// Fused multiply-add. Computes `(self * a) + b` with only one rounding
    /// error, yielding a more accurate result than an unfused multiply-add.
    ///
    /// Using `mul_add` *may* be more performant than an unfused multiply-add if
    /// the target architecture has a dedicated `fma` CPU instruction. However,
    /// this is not always true, and will be heavily dependant on designing
    /// algorithms with specific target hardware in mind.
    #[must_use = "method returns a new number and does not mutate the original value"]
    #[inline]
    fn mul_add(self, a: f32, b: f32) -> f32 {
        #[cfg(not(target_arch = "nvptx64"))]
        let val = self.mul_add(a, b);
        #[cfg(target_arch = "nvptx64")]
        let val = { unsafe { intrinsics::fmaf(self, a, b) } };
        val
    }

    /// Calculates Euclidean division, the matching method for `rem_euclid`.
    ///
    /// This computes the integer `n` such that
    /// `self = n * rhs + self.rem_euclid(rhs)`.
    /// In other words, the result is `self / rhs` rounded to the integer `n`
    /// such that `self >= n * rhs`.
    #[must_use = "method returns a new number and does not mutate the original value"]
    #[inline]
    fn div_euclid(self, rhs: f32) -> f32 {
        let q = (self / rhs).trunc();
        if self % rhs < 0.0 {
            return if rhs > 0.0 { q - 1.0 } else { q + 1.0 };
        }
        q
    }

    /// Calculates the least nonnegative remainder of `self (mod rhs)`.
    ///
    /// In particular, the return value `r` satisfies `0.0 <= r < rhs.abs()` in
    /// most cases. However, due to a floating point round-off error it can
    /// result in `r == rhs.abs()`, violating the mathematical definition, if
    /// `self` is much smaller than `rhs.abs()` in magnitude and `self < 0.0`.
    /// This result is not an element of the function's codomain, but it is the
    /// closest floating point number in the real numbers and thus fulfills the
    /// property `self == self.div_euclid(rhs) * rhs + self.rem_euclid(rhs)`
    /// approximatively.
    #[must_use = "method returns a new number and does not mutate the original value"]
    #[inline]
    fn rem_euclid(self, rhs: f32) -> f32 {
        let r = self % rhs;
        if r < 0.0 {
            r + rhs.abs()
        } else {
            r
        }
    }

    /// Raises a number to an integer power.
    ///
    /// Using this function is generally faster than using `powf`intrinsics
    #[must_use = "method returns a new number and does not mutate the original value"]
    #[inline]
    fn powi(self, n: i32) -> f32 {
        f32_intrinsic!(self, powi(n))
    }

    /// Raises a number to a floating point power.
    #[must_use = "method returns a new number and does not mutate the original value"]
    #[inline]
    fn powf(self, n: f32) -> f32 {
        #[cfg(not(target_arch = "nvptx64"))]
        let val = self.powf(n);
        #[cfg(target_arch = "nvptx64")]
        let val = { unsafe { intrinsics::powf(self, n) } };
        val
    }

    /// Returns the square root of a number.
    ///
    /// Returns NaN if `self` is a negative number other than `-0.0`.
    #[must_use = "method returns a new number and does not mutate the original value"]
    #[inline]
    fn sqrt(self) -> f32 {
        f32_intrinsic!(self, sqrt())
    }

    /// Returns `e^(self)`, (the exponential function).
    #[must_use = "method returns a new number and does not mutate the original value"]
    #[inline]
    fn exp(self) -> f32 {
        f32_intrinsic!(self, exp())
    }

    /// Returns `2^(self)`.
    #[must_use = "method returns a new number and does not mutate the original value"]
    #[inline]
    fn exp2(self) -> f32 {
        f32_intrinsic!(self, exp2())
    }

    /// Returns the natural logarithm of the number.
    #[must_use = "method returns a new number and does not mutate the original value"]
    #[inline]
    fn ln(self) -> f32 {
        #[cfg(not(target_arch = "nvptx64"))]
        let val = self.ln();
        #[cfg(target_arch = "nvptx64")]
        let val = { unsafe { intrinsics::logf(self) } };
        val
    }

    /// Returns the logarithm of the number with respect to an arbitrary base.
    ///
    /// The result might not be correctly rounded owing to implementation details;
    /// `self.log2()` can produce more accurate results for base 2, and
    /// `self.log10()` can produce more accurate results for base 10.
    #[must_use = "method returns a new number and does not mutate the original value"]
    #[inline]
    fn log(self, base: f32) -> f32 {
        self.ln() / base.ln()
    }

    /// Returns the base 2 logarithm of the number.
    #[must_use = "method returns a new number and does not mutate the original value"]
    #[inline]
    fn log2(self) -> f32 {
        f32_intrinsic!(self, log10())
    }

    /// Returns the base 10 logarithm of the number.
    #[must_use = "method returns a new number and does not mutate the original value"]
    #[inline]
    fn log10(self) -> f32 {
        f32_intrinsic!(self, log10())
    }

    /// Returns the cube root of a number.
    #[must_use = "method returns a new number and does not mutate the original value"]
    #[inline]
    fn cbrt(self) -> f32 {
        f32_intrinsic!(self, cbrt())
    }

    /// Calculates the length of the hypotenuse of a right-angle triangle given
    /// legs of length `x` and `y`.
    #[must_use = "method returns a new number and does not mutate the original value"]
    #[inline]
    fn hypot(self, other: f32) -> f32 {
        f32_intrinsic!(self, hypot(other))
    }

    /// Computes the sine of a number (in radians).
    #[must_use = "method returns a new number and does not mutate the original value"]
    #[inline]
    fn sin(self) -> f32 {
        f32_intrinsic!(self, sin())
    }

    /// Computes the cosine of a number (in radians).
    #[must_use = "method returns a new number and does not mutate the original value"]
    #[inline]
    fn cos(self) -> f32 {
        f32_intrinsic!(self, cos())
    }

    /// Computes the tangent of a number (in radians).
    #[must_use = "method returns a new number and does not mutate the original value"]
    #[inline]
    fn tan(self) -> f32 {
        f32_intrinsic!(self, tan())
    }

    /// Computes the arcsine of a number. Return value is in radians in
    /// the range [-pi/2, pi/2] or NaN if the number is outside the range
    /// [-1, 1].
    #[must_use = "method returns a new number and does not mutate the original value"]
    #[inline]
    fn asin(self) -> f32 {
        f32_intrinsic!(self, asin())
    }

    /// Computes the arccosine of a number. Return value is in radians in
    /// the range [0, pi] or NaN if the number is outside the range
    /// [-1, 1].
    #[must_use = "method returns a new number and does not mutate the original value"]
    #[inline]
    fn acos(self) -> f32 {
        f32_intrinsic!(self, acos())
    }

    /// Computes the arctangent of a number. Return value is in radians in the
    /// range [-pi/2, pi/2];intrinsics
    #[must_use = "method returns a new number and does not mutate the original value"]
    #[inline]
    fn atan(self) -> f32 {
        f32_intrinsic!(self, atan())
    }

    /// Computes the four quadrant arctangent of `self` (`y`) and `other` (`x`) in radians.
    ///
    /// * `x = 0`, `y = 0`: `0`
    /// * `x >= 0`: `arctan(y/x)` -> `[-pi/2, pi/2]`
    /// * `y >= 0`: `arctan(y/x) + pi` -> `(pi/2, pi]`
    /// * `y < 0`: `arctan(y/x) - pi` -> `(-pi, -pi/2)`intrinsics
    #[must_use = "method returns a new number and does not mutate the original value"]
    #[inline]
    fn atan2(self, other: f32) -> f32 {
        f32_intrinsic!(self, atan2(other))
    }

    /// Simultaneously computes the sine and cosine of the number, `x`. Returns
    /// `(sin(x), cos(x))`.
    #[inline]
    fn sin_cos(self) -> (f32, f32) {
        #[cfg(not(target_arch = "nvptx64"))]
        let val = self.sin_cos();
        #[cfg(target_arch = "nvptx64")]
        let val = {
            let mut sptr = 0.0;
            let mut cptr = 0.0;
            unsafe {
                intrinsics::sincosf(self, &mut sptr as *mut _, &mut cptr as *mut _);
            }
            (sptr, cptr)
        };
        val
    }

    /// Returns `e^(self) - 1` in a way that is accurate even if the
    /// number is close to zero.
    #[must_use = "method returns a new number and does not mutate the original value"]
    #[inline]
    fn exp_m1(self) -> f32 {
        #[cfg(not(target_arch = "nvptx64"))]
        let val = self.exp_m1();
        #[cfg(target_arch = "nvptx64")]
        let val = { unsafe { intrinsics::expm1f(self) } };
        val
    }

    /// Returns `ln(1+n)` (natural logarithm) more accurately than if
    /// the operations were performed separately.
    #[must_use = "method returns a new number and does not mutate the original value"]
    #[inline]
    fn ln_1p(self) -> f32 {
        #[cfg(not(target_arch = "nvptx64"))]
        let val = self.ln_1p();
        #[cfg(target_arch = "nvptx64")]
        let val = { unsafe { intrinsics::log1pf(self) } };
        val
    }

    /// Hyperbolic sine function.
    #[must_use = "method returns a new number and does not mutate the original value"]
    #[inline]
    fn sinh(self) -> f32 {
        f32_intrinsic!(self, sinh())
    }

    /// Hyperbolic cosine function.
    #[must_use = "method returns a new number and does not mutate the original value"]
    #[inline]
    fn cosh(self) -> f32 {
        f32_intrinsic!(self, cosh())
    }

    /// Hyperbolic tangent function.
    #[must_use = "method returns a new number and does not mutate the original value"]
    #[inline]
    fn tanh(self) -> f32 {
        f32_intrinsic!(self, tanh())
    }

    /// Inverse hyperbolic sine function.
    #[must_use = "method returns a new number and does not mutate the original value"]
    #[inline]
    fn asinh(self) -> f32 {
        f32_intrinsic!(self, asinh())
    }

    /// Inverse hyperbolic cosine function.
    #[must_use = "method returns a new number and does not mutate the original value"]
    #[inline]
    fn acosh(self) -> f32 {
        f32_intrinsic!(self, acosh())
    }

    /// Inverse hyperbolic tangent function.
    #[must_use = "method returns a new number and does not mutate the original value"]
    #[inline]
    fn atanh(self) -> f32 {
        f32_intrinsic!(self, atanh())
    }
}

impl GpuFloat for f64 {
    /// Returns the largest integer less than or equal to a number.
    #[must_use = "method returns a new number and does not mutate the original value"]
    #[inline]
    fn floor(self) -> f64 {
        f64_intrinsic!(self, floor())
    }

    /// Returns the smallest integer greater than or equal to a number.
    #[must_use = "method returns a new number and does not mutate the original value"]
    #[inline]
    fn ceil(self) -> f64 {
        f64_intrinsic!(self, ceil())
    }

    /// Returns the nearest integer to a number. Round half-way cases away from
    /// `0.0`.
    #[must_use = "method returns a new number and does not mutate the original value"]
    #[inline]
    fn round(self) -> f64 {
        f64_intrinsic!(self, round())
    }

    /// Returns the integer part of a number.
    #[must_use = "method returns a new number and does not mutate the original value"]
    #[inline]
    fn trunc(self) -> f64 {
        f64_intrinsic!(self, trunc())
    }

    /// Returns the fractional part of a number.
    #[must_use = "method returns a new number and does not mutate the original value"]
    #[inline]
    fn fract(self) -> f64 {
        self - self.trunc()
    }

    /// Computes the absolute value of `self`. Returns `NAN` if the
    /// number is `NAN`.
    #[must_use = "method returns a new number and does not mutate the original value"]
    #[inline]
    fn abs(self) -> f64 {
        #[cfg(not(target_arch = "nvptx64"))]
        let val = self.abs();
        #[cfg(target_arch = "nvptx64")]
        let val = { unsafe { intrinsics::fabs(self) } };
        val
    }

    /// Returns a number that represents the sign of `self`.
    ///
    /// - `1.0` if the number is positive, `+0.0` or `INFINITY`
    /// - `-1.0` if the number is negative, `-0.0` or `NEG_INFINITY`
    /// - `NAN` if the number is `NAN`intrinsics
    #[must_use = "method returns a new number and does not mutate the original value"]
    #[inline]
    fn signum(self) -> f64 {
        if self.is_nan() {
            Self::NAN
        } else {
            1.0_f64.copysign(self)
        }
    }

    /// Returns a number composed of the magnitude of `self` and the sign of
    /// `sign`.
    ///
    /// Equal to `self` if the sign of `self` and `sign` are the same, otherwise
    /// equal to `-self`. If `self` is a `NAN`, then a `NAN` with the sign of
    /// `sign` is returned.
    #[must_use = "method returns a new number and does not mutate the original value"]
    #[inline]
    fn copysign(self, sign: f64) -> f64 {
        f64_intrinsic!(self, copysign(sign))
    }

    /// Fused multiply-add. Computes `(self * a) + b` with only one rounding
    /// error, yielding a more accurate result than an unfused multiply-add.
    ///
    /// Using `mul_add` *may* be more performant than an unfused multiply-add if
    /// the target architecture has a dedicated `fma` CPU instruction. However,
    /// this is not always true, and will be heavily dependant on designing
    /// algorithms with specific target hardware in mind.
    #[must_use = "method returns a new number and does not mutate the original value"]
    #[inline]
    fn mul_add(self, a: f64, b: f64) -> f64 {
        #[cfg(not(target_arch = "nvptx64"))]
        let val = self.mul_add(a, b);
        #[cfg(target_arch = "nvptx64")]
        let val = { unsafe { intrinsics::fma(self, a, b) } };
        val
    }

    /// Calculates Euclidean division, the matching method for `rem_euclid`.
    ///
    /// This computes the integer `n` such that
    /// `self = n * rhs + self.rem_euclid(rhs)`.
    /// In other words, the result is `self / rhs` rounded to the integer `n`
    /// such that `self >= n * rhs`.
    #[must_use = "method returns a new number and does not mutate the original value"]
    #[inline]
    fn div_euclid(self, rhs: f64) -> f64 {
        let q = (self / rhs).trunc();
        if self % rhs < 0.0 {
            return if rhs > 0.0 { q - 1.0 } else { q + 1.0 };
        }
        q
    }

    /// Calculates the least nonnegative remainder of `self (mod rhs)`.
    ///
    /// In particular, the return value `r` satisfies `0.0 <= r < rhs.abs()` in
    /// most cases. However, due to a floating point round-off error it can
    /// result in `r == rhs.abs()`, violating the mathematical definition, if
    /// `self` is much smaller than `rhs.abs()` in magnitude and `self < 0.0`.
    /// This result is not an element of the function's codomain, but it is the
    /// closest floating point number in the real numbers and thus fulfills the
    /// property `self == self.div_euclid(rhs) * rhs + self.rem_euclid(rhs)`
    /// approximatively.
    #[must_use = "method returns a new number and does not mutate the original value"]
    #[inline]
    fn rem_euclid(self, rhs: f64) -> f64 {
        let r = self % rhs;
        if r < 0.0 {
            r + rhs.abs()
        } else {
            r
        }
    }

    /// Raises a number to an integer power.
    ///
    /// Using this function is generally faster than using `powf`intrinsics
    #[must_use = "method returns a new number and does not mutate the original value"]
    #[inline]
    fn powi(self, n: i32) -> f64 {
        f64_intrinsic!(self, powi(n))
    }

    /// Raises a number to a floating point power.
    #[must_use = "method returns a new number and does not mutate the original value"]
    #[inline]
    fn powf(self, n: f64) -> f64 {
        #[cfg(not(target_arch = "nvptx64"))]
        let val = self.powf(n);
        #[cfg(target_arch = "nvptx64")]
        let val = { unsafe { intrinsics::pow(self, n) } };
        val
    }

    /// Returns the square root of a number.
    ///
    /// Returns NaN if `self` is a negative number other than `-0.0`.
    #[must_use = "method returns a new number and does not mutate the original value"]
    #[inline]
    fn sqrt(self) -> f64 {
        f64_intrinsic!(self, sqrt())
    }

    /// Returns `e^(self)`, (the exponential function).
    #[must_use = "method returns a new number and does not mutate the original value"]
    #[inline]
    fn exp(self) -> f64 {
        f64_intrinsic!(self, exp())
    }

    /// Returns `2^(self)`.
    #[must_use = "method returns a new number and does not mutate the original value"]
    #[inline]
    fn exp2(self) -> f64 {
        f64_intrinsic!(self, exp2())
    }

    /// Returns the natural logarithm of the number.
    #[must_use = "method returns a new number and does not mutate the original value"]
    #[inline]
    fn ln(self) -> f64 {
        #[cfg(not(target_arch = "nvptx64"))]
        let val = self.ln();
        #[cfg(target_arch = "nvptx64")]
        let val = { unsafe { intrinsics::log(self) } };
        val
    }

    /// Returns the logarithm of the number with respect to an arbitrary base.
    ///
    /// The result might not be correctly rounded owing to implementation details;
    /// `self.log2()` can produce more accurate results for base 2, and
    /// `self.log10()` can produce more accurate results for base 10.
    #[must_use = "method returns a new number and does not mutate the original value"]
    #[inline]
    fn log(self, base: f64) -> f64 {
        self.ln() / base.ln()
    }

    /// Returns the base 2 logarithm of the number.
    #[must_use = "method returns a new number and does not mutate the original value"]
    #[inline]
    fn log2(self) -> f64 {
        f64_intrinsic!(self, log10())
    }

    /// Returns the base 10 logarithm of the number.
    #[must_use = "method returns a new number and does not mutate the original value"]
    #[inline]
    fn log10(self) -> f64 {
        f64_intrinsic!(self, log10())
    }

    /// Returns the cube root of a number.
    #[must_use = "method returns a new number and does not mutate the original value"]
    #[inline]
    fn cbrt(self) -> f64 {
        f64_intrinsic!(self, cbrt())
    }

    /// Calculates the length of the hypotenuse of a right-angle triangle given
    /// legs of length `x` and `y`.
    #[must_use = "method returns a new number and does not mutate the original value"]
    #[inline]
    fn hypot(self, other: f64) -> f64 {
        f64_intrinsic!(self, hypot(other))
    }

    /// Computes the sine of a number (in radians).
    #[must_use = "method returns a new number and does not mutate the original value"]
    #[inline]
    fn sin(self) -> f64 {
        f64_intrinsic!(self, sin())
    }

    /// Computes the cosine of a number (in radians).
    #[must_use = "method returns a new number and does not mutate the original value"]
    #[inline]
    fn cos(self) -> f64 {
        f64_intrinsic!(self, cos())
    }

    /// Computes the tangent of a number (in radians).
    #[must_use = "method returns a new number and does not mutate the original value"]
    #[inline]
    fn tan(self) -> f64 {
        f64_intrinsic!(self, tan())
    }

    /// Computes the arcsine of a number. Return value is in radians in
    /// the range [-pi/2, pi/2] or NaN if the number is outside the range
    /// [-1, 1].
    #[must_use = "method returns a new number and does not mutate the original value"]
    #[inline]
    fn asin(self) -> f64 {
        f64_intrinsic!(self, asin())
    }

    /// Computes the arccosine of a number. Return value is in radians in
    /// the range [0, pi] or NaN if the number is outside the range
    /// [-1, 1].
    #[must_use = "method returns a new number and does not mutate the original value"]
    #[inline]
    fn acos(self) -> f64 {
        f64_intrinsic!(self, acos())
    }

    /// Computes the arctangent of a number. Return value is in radians in the
    /// range [-pi/2, pi/2];intrinsics
    #[must_use = "method returns a new number and does not mutate the original value"]
    #[inline]
    fn atan(self) -> f64 {
        f64_intrinsic!(self, atan())
    }

    /// Computes the four quadrant arctangent of `self` (`y`) and `other` (`x`) in radians.
    ///
    /// * `x = 0`, `y = 0`: `0`
    /// * `x >= 0`: `arctan(y/x)` -> `[-pi/2, pi/2]`
    /// * `y >= 0`: `arctan(y/x) + pi` -> `(pi/2, pi]`
    /// * `y < 0`: `arctan(y/x) - pi` -> `(-pi, -pi/2)`intrinsics
    #[must_use = "method returns a new number and does not mutate the original value"]
    #[inline]
    fn atan2(self, other: f64) -> f64 {
        f64_intrinsic!(self, atan2(other))
    }

    /// Simultaneously computes the sine and cosine of the number, `x`. Returns
    /// `(sin(x), cos(x))`.
    #[inline]
    fn sin_cos(self) -> (f64, f64) {
        #[cfg(not(target_arch = "nvptx64"))]
        let val = self.sin_cos();
        #[cfg(target_arch = "nvptx64")]
        let val = {
            let mut sptr = 0.0;
            let mut cptr = 0.0;
            unsafe {
                intrinsics::sincos(self, &mut sptr as *mut _, &mut cptr as *mut _);
            }
            (sptr, cptr)
        };
        val
    }

    /// Returns `e^(self) - 1` in a way that is accurate even if the
    /// number is close to zero.
    #[must_use = "method returns a new number and does not mutate the original value"]
    #[inline]
    fn exp_m1(self) -> f64 {
        #[cfg(not(target_arch = "nvptx64"))]
        let val = self.exp_m1();
        #[cfg(target_arch = "nvptx64")]
        let val = { unsafe { intrinsics::expm1(self) } };
        val
    }

    /// Returns `ln(1+n)` (natural logarithm) more accurately than if
    /// the operations were performed separately.
    #[must_use = "method returns a new number and does not mutate the original value"]
    #[inline]
    fn ln_1p(self) -> f64 {
        #[cfg(not(target_arch = "nvptx64"))]
        let val = self.ln_1p();
        #[cfg(target_arch = "nvptx64")]
        let val = { unsafe { intrinsics::log1p(self) } };
        val
    }

    /// Hyperbolic sine function.
    #[must_use = "method returns a new number and does not mutate the original value"]
    #[inline]
    fn sinh(self) -> f64 {
        f64_intrinsic!(self, sinh())
    }

    /// Hyperbolic cosine function.
    #[must_use = "method returns a new number and does not mutate the original value"]
    #[inline]
    fn cosh(self) -> f64 {
        f64_intrinsic!(self, cosh())
    }

    /// Hyperbolic tangent function.
    #[must_use = "method returns a new number and does not mutate the original value"]
    #[inline]
    fn tanh(self) -> f64 {
        f64_intrinsic!(self, tanh())
    }

    /// Inverse hyperbolic sine function.
    #[must_use = "method returns a new number and does not mutate the original value"]
    #[inline]
    fn asinh(self) -> f64 {
        f64_intrinsic!(self, asinh())
    }

    /// Inverse hyperbolic cosine function.
    #[must_use = "method returns a new number and does not mutate the original value"]
    #[inline]
    fn acosh(self) -> f64 {
        f64_intrinsic!(self, acosh())
    }

    /// Inverse hyperbolic tangent function.
    #[must_use = "method returns a new number and does not mutate the original value"]
    #[inline]
    fn atanh(self) -> f64 {
        f64_intrinsic!(self, atanh())
    }
}
