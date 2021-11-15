#![cfg_attr(
    target_os = "cuda",
    no_std,
    feature(register_attr),
    register_attr(nvvm_internal)
)]
#![allow(clippy::missing_safety_doc)]

extern crate alloc;

pub mod hittable;
pub mod material;
pub mod math;
pub mod render;
pub mod render_kernels;
pub mod scene;
pub mod sphere;

pub use cuda_std::vek;
use enum_dispatch::enum_dispatch;
use hittable::{HitRecord, Hittable};
use sphere::Sphere;

pub type Vec3 = vek::Vec3<f32>;
pub type Point = vek::Vec3<f32>;
pub type Vec2 = vek::Vec2<f32>;

#[cfg_attr(not(target_os = "cuda"), derive(cust::DeviceCopy))]
#[derive(Default, Clone, Copy)]
#[repr(C)]
pub struct Viewport {
    pub bounds: vek::Vec2<usize>,
    pub lower_left: Vec3,
    pub horizontal: Vec3,
    pub vertical: Vec3,
    pub origin: Vec3,
}

#[cfg_attr(not(target_os = "cuda"), derive(cust::DeviceCopy))]
#[repr(C)]
#[derive(Clone, Copy)]
#[enum_dispatch(Hittable)]
pub enum Object {
    Sphere(Sphere),
}

#[derive(Clone, Copy, PartialEq)]
pub struct Ray {
    pub dir: Vec3,
    pub origin: Point,
}

impl Ray {
    pub fn new(dir: Vec3, origin: Point) -> Self {
        Self { dir, origin }
    }

    pub fn at(&self, t: f32) -> Point {
        self.origin + t * self.dir
    }
}
