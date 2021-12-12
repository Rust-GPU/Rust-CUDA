use gpu_rand::DefaultRand;

use crate::material::*;
use crate::*;

const MAX_BOUNCES: u32 = 5;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Scene<'a> {
    pub objects: &'a [Object],
    pub materials: &'a [MaterialKind],
}

/// SAFETY: the slice is created from unified memory so it works on the GPU too.
#[cfg(not(target_os = "cuda"))]
unsafe impl cust::memory::DeviceCopy for Scene<'_> {}

impl Scene<'_> {
    pub fn hit(&self, ray: Ray, t_min: f32, t_max: f32) -> Option<HitRecord> {
        let mut hit = None;
        let mut closest_so_far = t_max;

        for obj in self.objects {
            if let Some(rec) = obj.hit(ray, t_min, closest_so_far) {
                hit = Some(rec);
                closest_so_far = rec.t;
            }
        }
        hit
    }

    /// Casts a ray into the scene and returns the object hit by the ray.
    pub fn raycast(&self, ray: Ray) -> Option<&Object> {
        let mut hit = None;
        let mut closest_so_far = f32::INFINITY;

        for obj in self.objects {
            if let Some(rec) = obj.hit(ray, 0.001, closest_so_far) {
                hit = Some(obj);
                closest_so_far = rec.t;
            }
        }
        hit
    }

    pub fn ray_color(&self, ray: Ray, rng: &mut DefaultRand) -> Vec3 {
        let mut cur_ray = ray;
        let mut attenuation = Vec3::one();

        for _ in 0..MAX_BOUNCES {
            if let Some(hit) = self.hit(cur_ray, 0.001, f32::INFINITY) {
                let material = self.materials[hit.material_handle];
                let (hit_attenuation, scattered) = material.scatter(cur_ray, hit, rng);
                if let Some(scattered) = scattered {
                    attenuation *= hit_attenuation;
                    cur_ray = scattered;
                } else {
                    return Vec3::zero();
                }
            } else {
                let unit = cur_ray.dir.normalized();
                let t = 0.5 * (unit.y + 1.0);
                let c = (1.0 - t) * Vec3::one() + t * Vec3::new(0.5, 0.7, 1.0);
                return attenuation * c;
            }
        }
        Vec3::zero()
    }
}
