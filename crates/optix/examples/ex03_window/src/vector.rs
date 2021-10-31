use core::ops;
pub use num_traits::{One, Zero};

pub trait Scalar: num_traits::One + num_traits::Zero {}

impl Scalar for i8 {}
impl Scalar for i16 {}
impl Scalar for i32 {}
impl Scalar for i64 {}
impl Scalar for f32 {}
impl Scalar for f64 {}

pub trait Vector {
    type Component: Scalar;

    fn dot(&self, v: &Self) -> Self::Component;

    #[inline]
    fn length2(&self) -> Self::Component {
        self.dot(&self)
    }
}

macro_rules! vec_impl {
    ($name:ident: $t:ty, $sc:ident, $align:expr, ($($c:ident),+)) => {
        #[repr(C)]
        #[derive(Clone, Copy, Default, PartialEq, Debug)]
        pub struct $name
        {
            $(
                pub $c: $t,
            )+
        }

        impl $name
        {
            pub fn new($($c: $t),+) -> Self
            {
                Self {
                    $(
                        $c,
                    )+
                }
            }
        }

        impl Vector for $name
        {
            type Component = $t;

            #[inline]
            fn dot(&self, v: &Self) -> $t
            {
                <$t>::zero() $(
                    + self.$c * v.$c
                )+
            }
        }

        impl From<$t> for $name
        {
            fn from(x: $t) -> Self
            {
                Self {
                    $(
                        $c: x,
                    )+
                }
            }
        }

        impl ops::Neg for $name
        {
            type Output = Self;

            fn neg(self) -> Self
            {
                Self {
                    $(
                        $c: -self.$c,
                    )+
                }
            }
        }

        impl ops::Add for $name
        {
            type Output = Self;

            #[inline]
            fn add(self, v: Self) -> Self
            {
                Self {
                    $(
                        $c: self.$c + v.$c,
                    )+
                }
            }
        }

        impl ops::AddAssign for $name
        {
            #[inline]
            fn add_assign(&mut self, v: Self)
            {
                $(
                    self.$c += v.$c;
                )+
            }
        }

        impl ops::Sub for $name
        {
            type Output = Self;

            #[inline]
            fn sub(self, v: Self) -> Self
            {
                Self {
                    $(
                        $c: self.$c - v.$c,
                    )+
                }
            }
        }

        impl ops::SubAssign for $name
        {
            #[inline]
            fn sub_assign(&mut self, v: Self)
            {
                $(
                    self.$c -= v.$c;
                )+
            }
        }

        impl ops::Mul for $name
        {
            type Output = Self;

            #[inline]
            fn mul(self, v: Self) -> Self
            {
                Self {
                    $(
                        $c: self.$c * v.$c,
                    )+
                }
            }
        }

        impl ops::MulAssign for $name
        {
            #[inline]
            fn mul_assign(&mut self, v: Self)
            {
                $(
                    self.$c *= v.$c;
                )+
            }
        }

        impl ops::Mul<$t> for $name
        {
            type Output = Self;

            #[inline]
            fn mul(self, v: $t) -> Self
            {
                Self {
                    $(
                        $c: self.$c * v,
                    )+
                }
            }
        }

        impl ops::MulAssign<$t> for $name
        {
            #[inline]
            fn mul_assign(&mut self, v: $t)
            {
                $(
                    self.$c *= v;
                )+
            }
        }

        impl ops::Div<$t> for $name
        {
            type Output = Self;

            #[inline]
            fn div(self, v: $t) -> Self
            {
                Self {
                    $(
                        $c: self.$c / v,
                    )+
                }
            }
        }

        impl ops::DivAssign<$t> for $name
        {
            #[inline]
            fn div_assign(&mut self, v: $t)
            {
                $(
                    self.$c /= v;
                )+
            }
        }

        impl ops::Mul<$name> for $t
        {
            type Output = $name;

            #[inline]
            fn mul(self, v: $name) -> $name
            {
                $name {
                    $(
                        $c: self * v.$c,
                    )+
                }
            }
        }

        impl ops::Div<$name> for $t
        {
            type Output = $name;

            #[inline]
            fn div(self, v: $name) -> $name
            {
                $name {
                    $(
                        $c: self / v.$c,
                    )+
                }
            }
        }

        pub fn $sc($($c: $t),+) -> $name
        {
            $name {
                $(
                    $c,
                )+
            }
        }

        unsafe impl cust::memory::DeviceCopy for $name {
            // fn device_align() -> usize {
            //     $align
            // }
        }
    };

}

vec_impl!(V2i8: i8, v2i8, 1, (x, y));
vec_impl!(V2i16: i16, v2i16, 2, (x, y));
vec_impl!(V2i32: i32, v2i32, 8, (x, y));
vec_impl!(V2i64: i64, v2i64, 8, (x, y));
vec_impl!(V3i8: i8, v3i8, 1, (x, y, z));
vec_impl!(V3i16: i16, v3i16, 2, (x, y, z));
vec_impl!(V3i32: i32, v3i32, 4, (x, y, z));
vec_impl!(V3i64: i64, v3i64, 8, (x, y, z));
vec_impl!(V4i8: i8, v4i8, 1, (x, y, z, w));
vec_impl!(V4i16: i16, v4i16, 2, (x, y, z, w));
vec_impl!(V4i32: i32, v4i32, 16, (x, y, z, w));
vec_impl!(V4i64: i64, v4i64, 8, (x, y, z, w));

vec_impl!(V2f32: f32, v2f32, 8, (x, y));
vec_impl!(V2f64: f64, v2f64, 8, (x, y));
vec_impl!(V3f32: f32, v3f32, 4, (x, y, z));
vec_impl!(V3f64: f64, v3f64, 8, (x, y, z));
vec_impl!(V4f32: f32, v4f32, 16, (x, y, z, w));
vec_impl!(V4f64: f64, v4f64, 8, (x, y, z, w));

vec_impl!(P2f32: f32, p2f32, 8, (x, y));
vec_impl!(P2f64: f64, p2f64, 8, (x, y));
vec_impl!(P3f32: f32, p3f32, 4, (x, y, z));
vec_impl!(P3f64: f64, p3f64, 8, (x, y, z));
vec_impl!(P4f32: f32, p4f32, 16, (x, y, z, w));
vec_impl!(P4f64: f64, p4f64, 8, (x, y, z, w));

vec_impl!(N2f32: f32, n2f32, 8, (x, y));
vec_impl!(N2f64: f64, n2f64, 8, (x, y));
vec_impl!(N3f32: f32, n3f32, 4, (x, y, z));
vec_impl!(N3f64: f64, n3f64, 8, (x, y, z));
vec_impl!(N4f32: f32, n4f32, 16, (x, y, z, w));
vec_impl!(N4f64: f64, n4f64, 8, (x, y, z, w));

#[inline]
pub fn dot<T: Vector>(a: &T, b: &T) -> T::Component {
    a.dot(b)
}
