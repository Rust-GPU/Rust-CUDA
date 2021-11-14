//! Generic math utilities.

use crate::Vec3;
#[cfg(target_os = "cuda")]
use cuda_std::GpuFloat;
use gpu_rand::{DefaultRand, GpuRand};

/// Converts a float in the range of [0.0, 1.0] to a range of [-1.0, 1.0].
pub fn norm_f32_to_snorm(x: f32) -> f32 {
    x * 2.0 - 1.0
}

pub fn random_unit_vec(state: &mut DefaultRand) -> Vec3 {
    let [x, y] = state.normal_f32_2();
    let z = state.normal_f32();
    Vec3::new(x, y, z)
}

/// Creates a random vector with each element being in the range of [-1.0, 1.0] (signed normalized).
pub fn random_snorm_vec(state: &mut DefaultRand) -> Vec3 {
    random_unit_vec(state).map(norm_f32_to_snorm)
}

pub fn random_in_unit_sphere(state: &mut DefaultRand) -> Vec3 {
    loop {
        let p = random_snorm_vec(state);
        if p.magnitude_squared() >= 1.0 {
            continue;
        }
        return p;
    }
}

pub fn reflect(v: Vec3, n: Vec3) -> Vec3 {
    v - 2.0 * v.dot(n) * n
}

pub fn refract(uv: Vec3, n: Vec3, etai_over_etat: f32) -> Vec3 {
    let cos_t = (-uv).dot(n).min(1.0);
    let r_out_perp = etai_over_etat * (uv + cos_t * n);
    let r_out_parallel = -((1.0 - r_out_perp.magnitude_squared()).abs()).sqrt() * n;
    r_out_perp + r_out_parallel
}
