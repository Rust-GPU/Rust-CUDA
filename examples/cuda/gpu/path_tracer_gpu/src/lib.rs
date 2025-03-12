#![cfg_attr(target_os = "cuda", no_std, register_attr(nvvm_internal))]
#![allow(clippy::missing_safety_doc)]

extern crate alloc;

pub mod hittable;
pub mod material;
pub mod math;
pub mod optix;
pub mod render;
pub mod render_kernels;
pub mod scene;
pub mod sphere;

pub use cuda_std::vek;
use cust_core::DeviceCopy;
use enum_dispatch::enum_dispatch;
use hittable::{HitRecord, Hittable};
use sphere::Sphere;

pub type Vec3<T = f32> = vek::Vec3<T>;
pub type Point<T = f32> = vek::Vec3<T>;
pub type Vec2<T = f32> = vek::Vec2<T>;

#[derive(Default, Clone, Copy, DeviceCopy)]
#[repr(C)]
pub struct Viewport {
    pub bounds: vek::Vec2<usize>,
    pub lower_left: Vec3,
    pub horizontal: Vec3,
    pub vertical: Vec3,
    pub origin: Vec3,
}

#[repr(C)]
#[derive(Clone, Copy, DeviceCopy)]
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

    pub fn from_optix() -> Self {
        use optix_device::intersection;

        Self {
            dir: Vec3::from(intersection::ray_world_direction().to_array()),
            origin: Vec3::from(intersection::ray_world_origin().to_array()),
        }
    }
}
