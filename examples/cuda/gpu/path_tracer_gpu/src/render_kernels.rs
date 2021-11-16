use crate::{render::*, scene::Scene, *};
use cuda_std::{vek::Clamp, *};
use gpu_rand::{DefaultRand, GpuRand};

#[kernel]
pub unsafe fn render(fb: *mut Vec3, view: Viewport, scene: &Scene, rand_states: *mut DefaultRand) {
    let idx = thread::index_2d();
    if idx.x >= view.bounds.x as u32 || idx.y >= view.bounds.y as u32 {
        return;
    }
    let px_idx = idx.y as usize * view.bounds.x + idx.x as usize;

    // generate a tiny offset for the ray for antialiasing
    let rng = &mut *rand_states.add(px_idx);
    let offset = Vec2::from(rng.normal_f32_2());

    let ray = generate_ray(idx, &view, offset);

    let color = scene.ray_color(ray, rng);
    *fb.add(px_idx) += color;
}

/// Scales an accumulated buffer by the sample count, storing each pixel in the corresponding `out` pixel.
#[kernel]
pub unsafe fn scale_buffer(fb: *const Vec3, out: *mut Vec3, samples: u32, view: Viewport) {
    let idx_2d = thread::index_2d();
    if idx_2d.x >= view.bounds.x as u32 || idx_2d.y >= view.bounds.y as u32 {
        return;
    }
    let idx = idx_2d.y as usize * view.bounds.x + idx_2d.x as usize;
    let original = &*fb.add(idx);
    let out = &mut *out.add(idx);

    let scale = 1.0 / samples as f32;
    let scaled = original * scale;
    *out = scaled;
}

/// Postprocesses a (scaled) buffer into a final u8 buffer.
#[kernel]
pub unsafe fn postprocess(fb: *const Vec3, out: *mut vek::Vec3<u8>, view: Viewport) {
    let idx_2d = thread::index_2d();
    if idx_2d.x >= view.bounds.x as u32 || idx_2d.y >= view.bounds.y as u32 {
        return;
    }
    let idx = idx_2d.y as usize * view.bounds.x + idx_2d.x as usize;
    let original = &*fb.add(idx);
    let out = &mut *out.add(idx);
    // gamma=2.0
    let gamma_corrected = original.sqrt();

    *out = (gamma_corrected * 255.0)
        .clamped(Vec3::zero(), Vec3::broadcast(255.0))
        .numcast()
        .unwrap();
}
