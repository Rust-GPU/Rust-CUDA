use crate::{
    hittable::HitRecord,
    math::{random_in_unit_sphere, reflect},
    Ray, Vec3,
};
use enum_dispatch::enum_dispatch;
use gpu_rand::DefaultRand;

#[enum_dispatch]
pub trait Material {
    /// Optionally scatters a ray and returns an attenuation color and an optional ray
    fn scatter(&self, incoming: Ray, hit: HitRecord, rng: &mut DefaultRand) -> (Vec3, Option<Ray>);
}

#[derive(Clone, Copy, PartialEq)]
#[cfg_attr(not(target_os = "cuda"), derive(cust::DeviceCopy))]
#[enum_dispatch(Material)]
pub enum MaterialKind {
    Diffuse(DiffuseMaterial),
    Metallic(MetallicMaterial),
}

#[derive(Clone, Copy, PartialEq)]
#[cfg_attr(not(target_os = "cuda"), derive(cust::DeviceCopy))]
pub struct DiffuseMaterial {
    pub color: Vec3,
}

impl Material for DiffuseMaterial {
    fn scatter(&self, _: Ray, hit: HitRecord, rng: &mut DefaultRand) -> (Vec3, Option<Ray>) {
        let mut scatter_dir = hit.normal + random_in_unit_sphere(rng);

        if scatter_dir.is_approx_zero() {
            scatter_dir = hit.normal;
        }

        let attenuation = self.color;
        let ray = Ray {
            origin: hit.point,
            dir: scatter_dir,
        };
        (attenuation, Some(ray))
    }
}

#[derive(Clone, Copy, PartialEq)]
#[cfg_attr(not(target_os = "cuda"), derive(cust::DeviceCopy))]
pub struct MetallicMaterial {
    pub color: Vec3,
    pub roughness: f32,
}

impl Material for MetallicMaterial {
    fn scatter(&self, incoming: Ray, hit: HitRecord, rng: &mut DefaultRand) -> (Vec3, Option<Ray>) {
        let reflected = reflect(incoming.dir.normalized(), hit.normal);
        let scattered = Ray {
            origin: hit.point,
            dir: reflected + self.roughness * random_in_unit_sphere(rng),
        };
        let attenuation = self.color;
        if scattered.dir.dot(hit.normal) > 0.0 {
            (attenuation, Some(scattered))
        } else {
            (attenuation, None)
        }
    }
}
