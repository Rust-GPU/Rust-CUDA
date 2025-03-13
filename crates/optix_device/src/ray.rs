use crate::trace::*;
#[cfg(target_os = "cuda")]
use core::arch::asm;
use cuda_std::gpu_only;
use glam::Vec3;

/// Returns the ray origin that was passed into [`trace`] in world-space.
///
/// May be more expensive to call in intersection and anyhit programs than it's object-space
/// counterpart. So the object-space variant should be used instead if possible.
#[gpu_only]
pub fn ray_world_origin() -> Vec3 {
    let x: f32;
    let y: f32;
    let z: f32;

    unsafe {
        asm!("call ({}), _optix_get_world_ray_origin_x, ();", out(reg32) x);
        asm!("call ({}), _optix_get_world_ray_origin_y, ();", out(reg32) y);
        asm!("call ({}), _optix_get_world_ray_origin_z, ();", out(reg32) z);
    }

    Vec3::new(x, y, z)
}

/// Returns the ray direction that was passed into [`trace`] in world-space.
///
/// May be more expensive to call in intersection and anyhit programs than it's object-space
/// counterpart. So the object-space variant should be used instead if possible.
#[gpu_only]
pub fn ray_world_direction() -> Vec3 {
    let x: f32;
    let y: f32;
    let z: f32;

    unsafe {
        asm!("call ({}), _optix_get_world_ray_direction_x, ();", out(reg32) x);
        asm!("call ({}), _optix_get_world_ray_direction_y, ();", out(reg32) y);
        asm!("call ({}), _optix_get_world_ray_direction_z, ();", out(reg32) z);
    }

    Vec3::new(x, y, z)
}

/// Returns the ray tmin that was passed into [`trace`].
#[gpu_only]
pub fn ray_tmin() -> f32 {
    let x: f32;

    unsafe {
        asm!(
            "{{",
            ".reg .f32 %f<1>;",
            "call (%f0), _optix_get_ray_tmin, ();",
            "mov.f32 {}, %f0;",
            "}}",
            out(reg32) x
        );
    }

    x
}

/// Returns the ray tmax that was passed into [`trace`].
#[gpu_only]
pub fn ray_tmax() -> f32 {
    let x: f32;

    unsafe {
        asm!(
            "{{",
            ".reg .f32 %f<1>;",
            "call (%f0), _optix_get_ray_tmax, ();",
            "mov.f32 {}, %f0;",
            "}}",
            out(reg32) x
        );
    }

    x
}

/// Returns the ray time that was passed into [`trace`].
#[gpu_only]
pub fn ray_time() -> f32 {
    let x: f32;

    unsafe {
        asm!("call ({}), _optix_get_ray_time, ();", out(reg32) x);
    }

    x
}

/// Returns the ray flags that were passed into [`trace`].
#[gpu_only]
pub fn ray_flags() -> RayFlags {
    let x: u32;

    unsafe {
        asm!("call ({}), _optix_get_ray_flags, ();", out(reg32) x);
    }

    RayFlags::from_bits_truncate(x)
}

/// Returns the ray visibility mask that was passed into [`trace`].
#[gpu_only]
pub fn ray_visibility_mask() -> u8 {
    let x: u32;

    unsafe {
        asm!("call ({}), _optix_get_ray_visibility_mask, ();", out(reg32) x);
    }

    x as u8
}

// -------------------- intersection and anyhit-specific

/// Returns the ray's object-space origin based on the current transform stack.
#[gpu_only]
pub fn ray_object_origin() -> Vec3 {
    let x: f32;
    let y: f32;
    let z: f32;

    unsafe {
        asm!("call ({}), _optix_get_object_ray_origin_x, ();", out(reg32) x);
        asm!("call ({}), _optix_get_object_ray_origin_y, ();", out(reg32) y);
        asm!("call ({}), _optix_get_object_ray_origin_z, ();", out(reg32) z);
    }

    Vec3::new(x, y, z)
}

/// Returns the ray's object-space direction based on the current transform stack.
#[gpu_only]
pub fn ray_object_direction() -> Vec3 {
    let x: f32;
    let y: f32;
    let z: f32;

    unsafe {
        asm!("call ({}), _optix_get_object_ray_direction_x, ();", out(reg32) x);
        asm!("call ({}), _optix_get_object_ray_direction_y, ();", out(reg32) y);
        asm!("call ({}), _optix_get_object_ray_direction_z, ();", out(reg32) z);
    }

    Vec3::new(x, y, z)
}
