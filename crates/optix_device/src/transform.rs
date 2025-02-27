// use std::hint::unreachable_unchecked;
use cuda_std::gpu_only;
use glam::{Vec3, Vec4};

use crate::{intersection::ray_time, trace::TraversableHandle};

#[gpu_only]
fn transform_list_size() -> u32 {
    let size: u32;
    unsafe {
        asm!("call ({}), _optix_get_transform_list_size, ();", out(reg32) size);
    }
    size
}

#[gpu_only]
fn transform_list_handle(i: u32) -> TraversableHandle {
    let handle: u64;
    unsafe {
        asm!("call ({}), _optix_get_transform_list_handle, ({});", out(reg64) handle, in(reg32) i);
    }
    TraversableHandle(handle)
}

#[gpu_only]
fn transform_type_from_handle(handle: TraversableHandle) -> TransformType {
    let type_: u32;
    unsafe {
        asm!("call ({}), _optix_get_transform_type_from_handle, ({});", out(reg32) type_, in(reg64) handle.0);
        core::mem::transmute(type_)
    }
}

/// The different types of transformations.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TransformType {
    None = 0,
    StaticTransform = 1,
    MatrixMotionTransform = 2,
    SrtMotionTransform = 3,
    Instance = 4,
}

// just to make sure rust actually aligns this, though the padding should already
// do this.
#[repr(C, align(16))]
#[derive(Debug, Clone, PartialEq)]
pub struct MatrixMotionTransform {
    pub child: TraversableHandle,
    pub motion_options: MotionOptions,
    /// Padding to make the struct 16-byte aligned.
    _pad: [u32; 3],
    /// two motion keys as a 3x4 object -> world transformation matrix in row-major order.
    pub transform: [[Vec4; 4]; 2],
}

/// Returns a reference to the matrix motion transfer of a traversable handle. Returns `None`
/// if the handle is not a valid handle for a matrix motion transfer.
#[gpu_only]
pub fn matrix_motion_transform_from_handle(
    handle: TraversableHandle,
) -> Option<&'static MatrixMotionTransform> {
    let transform_ptr: *const MatrixMotionTransform;
    unsafe {
        asm!("call ({}), _optix_get_matrix_motion_transform_from_handle, ({});",
            out(reg64) transform_ptr,
            in(reg64) handle.0
        );
        // SAFETY: according to the optix docs this function returns 0 if the handle is not valid.
        // And Option<&'static T> is the same as *const T because of NPO. And finally, this transform is valid
        // for the duration of the optix invocation, which is 'static as far as rust is concerned.
        core::mem::transmute(transform_ptr)
    }
}

// this is kind of cursed but we need to use a pointer instead of a ref so we can
// read past the end of the struct to read the rest of the motion keys if needed.
// refs dont allow reading past the end of the struct because of provenance.
#[gpu_only]
fn matrix_motion_transform_from_handle_ptr(
    handle: TraversableHandle,
) -> *const MatrixMotionTransform {
    let transform_ptr: *const MatrixMotionTransform;

    unsafe {
        asm!("call ({}), _optix_get_matrix_motion_transform_from_handle, ({});",
            out(reg64) transform_ptr,
            in(reg64) handle.0
        );
    }

    transform_ptr
}

bitflags::bitflags! {
    /// Possible motion flags.
    #[repr(transparent)]
    #[derive(Debug, Clone, PartialEq)]
    pub struct MotionFlags: u32 {
        const START_VANISH = 1 << 0;
        const END_VANISH   = 1 << 1;
    }
}

/// Options for motion.
#[repr(C)]
#[derive(Debug, Clone, PartialEq)]
pub struct MotionOptions {
    /// The number of motion keys. If the key count is over `1`, motion is enabled.
    /// Otherwise the rest of the fields are ignored.
    pub keys: u16,
    /// Possible motion flags.
    pub flags: MotionFlags,
    /// The start time of the motion.
    pub time_begin: f32,
    /// The end time of the motion.
    pub time_end: f32,
}

// fn resolve_motion_key(options: MotionOptions, global_t: f32) -> (f32, u32) {
//     let intervals = options.keys - 1;

//     let time = (intervals as f32)
//         .min(
//             ((global_t - options.time_begin) * intervals as f32)
//                 / (options.time_end - options.time_begin),
//         )
//         .max(0.0);
//     let flt_key = time.floor();

//     ((time - flt_key), flt_key as u32)
// }

// #[gpu_only]
// fn readonly_load_matrix(ptr: *const [Vec4; 3]) -> [Vec4; 3] {
//     let mut out = [Vec4::ZERO; 3];
//     for i in 0..3 {
//         let row = &mut out[i];
//         let row_ptr: *const Vec4;
//         unsafe {
//             asm!(
//                 "cvta.to.global.u64 {}, {};",
//                 out(reg64) row_ptr,
//                 in(reg64) ptr.add(i)
//             );
//             asm!(
//                 "ld.global.v4.u32 {{{}, {}, {}, {}}}, [{}];",
//                 out(reg32) row.x,
//                 out(reg32) row.y,
//                 out(reg32) row.z,
//                 out(reg32) row.w,
//                 in(reg64) row_ptr
//             );
//         }
//     }
//     out
// }

// fn interpolated_matrix_motion_transform(
//     transform: *const MatrixMotionTransform,
//     time: f32,
// ) -> [Vec4; 3] {
//     let (key_time, key) = resolve_motion_key(transform.motion_options, time);

//     unsafe {
//         // motion keys are unbelievably cursed, they store the first two keys
//         // then the rest are past the end of the struct.
//         let base = core::ptr::addr_of!((*transform).transform).cast::<[Vec4; 4]>();
//         let key_ptr = base.add(key);
//     }
// }

// fn interpolate_matrix_key(mat: *const [Vec4; 3], t1: f32) -> [Vec4; 3] {
//     let mut out = readonly_load_matrix(mat);
//     if t1 > 0.0 {
//         let t0 = 1.0 - t1;
//         out = (out * t0)
//     }
// }

#[repr(C)]
#[derive(Debug, Clone, PartialEq)]
pub struct StaticTransform {
    pub child: TraversableHandle,
    _pad: [u32; 2],
    pub transform: [Vec4; 3],
    pub inverse: [Vec4; 3],
}

#[gpu_only]
pub fn instance_transform_from_handle(handle: TraversableHandle) -> Option<&'static [Vec4; 3]> {
    let transform_ptr: *const [Vec4; 3];
    unsafe {
        asm!("call ({}), _optix_get_instance_transform_from_handle, ({});",
            out(reg64) transform_ptr,
            in(reg64) handle.0
        );
        // SAFETY: we know this reference will be valid for the duration of the optix invocation
        // and the repr is right because of NPO.
        core::mem::transmute(transform_ptr)
    }
}

#[gpu_only]
pub fn instance_inverse_transform_from_handle(
    handle: TraversableHandle,
) -> Option<&'static [Vec4; 3]> {
    let transform_ptr: *const [Vec4; 3];
    unsafe {
        asm!("call ({}), _optix_get_instance_inverse_transform_from_handle, ({});",
            out(reg64) transform_ptr,
            in(reg64) handle.0
        );
        // SAFETY: we know this reference will be valid for the duration of the optix invocation
        // and the repr is right because of NPO.
        core::mem::transmute(transform_ptr)
    }
}

#[gpu_only]
pub fn static_transform_from_handle(handle: TraversableHandle) -> Option<&'static StaticTransform> {
    let transform_ptr: *const StaticTransform;
    unsafe {
        asm!("call ({}), _optix_get_static_transform_from_handle, ({});",
            out(reg64) transform_ptr,
            in(reg64) handle.0
        );
        // SAFETY: we know this reference will be valid for the duration of the optix invocation
        // and the repr is right because of NPO.
        core::mem::transmute(transform_ptr)
    }
}

fn identity() -> [Vec4; 3] {
    [
        Vec4::new(1.0, 0.0, 0.0, 0.0),
        Vec4::new(0.0, 1.0, 0.0, 0.0),
        Vec4::new(0.0, 0.0, 1.0, 0.0),
    ]
}

fn interpolated_transformation(
    handle: TraversableHandle,
    _time: f32,
    object_to_world: bool,
) -> [Vec4; 3] {
    let ty = transform_type_from_handle(handle);

    if ty == TransformType::MatrixMotionTransform || ty == TransformType::SrtMotionTransform {
        todo!();
    } else if ty == TransformType::StaticTransform || ty == TransformType::Instance {
        let transform = if ty == TransformType::Instance {
            if object_to_world {
                instance_transform_from_handle(handle).unwrap()
            } else {
                instance_inverse_transform_from_handle(handle).unwrap()
            }
        } else {
            let trf = static_transform_from_handle(handle).unwrap();
            if object_to_world {
                &trf.transform
            } else {
                &trf.inverse
            }
        };
        *transform
    } else {
        identity()
    }
}

fn multiply_row_matrix(vec: Vec4, mat: &[Vec4; 3]) -> Vec4 {
    let mut out = Vec4::ZERO;
    out.x = vec.x * mat[0].x + vec.y * mat[1].x + vec.z * mat[2].x;
    out.y = vec.x * mat[0].y + vec.y * mat[1].y + vec.z * mat[2].y;
    out.z = vec.x * mat[0].z + vec.y * mat[1].z + vec.z * mat[2].z;
    out.w = vec.x * mat[0].w + vec.y * mat[1].w + vec.z * mat[2].w + vec.w;
    out
}

pub fn world_to_object_transform_matrix() -> [Vec4; 3] {
    let size = transform_list_size();
    let time = ray_time();
    let mut out = identity();

    if size == 0 {
        out
    } else {
        for i in 0..size {
            let handle = transform_list_handle(i);
            let mat = interpolated_transformation(handle, time, false);
            let tmp = out;
            out[0] = multiply_row_matrix(mat[0], &tmp);
            out[1] = multiply_row_matrix(mat[1], &tmp);
            out[2] = multiply_row_matrix(mat[2], &tmp);
        }
        out
    }
}

pub fn object_to_world_transform_matrix() -> [Vec4; 3] {
    let size = transform_list_size();
    let time = ray_time();
    let mut out = identity();

    if size == 0 {
        out
    } else {
        for i in (0..size).rev() {
            let handle = transform_list_handle(i);
            let mat = interpolated_transformation(handle, time, true);
            let tmp = out;
            out[0] = multiply_row_matrix(mat[0], &tmp);
            out[1] = multiply_row_matrix(mat[1], &tmp);
            out[2] = multiply_row_matrix(mat[2], &tmp);
        }
        out
    }
}

pub fn transform_point(mat: &[Vec4; 3], p: Vec3) -> Vec3 {
    let mut out = Vec3::ZERO;
    out.x = mat[0].x * p.x + mat[0].y * p.y + mat[0].z * p.z + mat[0].w;
    out.y = mat[1].x * p.x + mat[1].y * p.y + mat[1].z * p.z + mat[1].w;
    out.z = mat[2].x * p.x + mat[2].y * p.y + mat[2].z * p.z + mat[2].w;
    out
}

pub fn transform_vector(mat: &[Vec4; 3], v: Vec3) -> Vec3 {
    let mut out = Vec3::ZERO;
    out.x = mat[0].x * v.x + mat[0].y * v.y + mat[0].z * v.z;
    out.y = mat[1].x * v.x + mat[1].y * v.y + mat[1].z * v.z;
    out.z = mat[2].x * v.x + mat[2].y * v.y + mat[2].z * v.z;
    out
}

pub fn transform_normal(mat: &[Vec4; 3], n: Vec3) -> Vec3 {
    let mut out = Vec3::ZERO;
    out.x = mat[0].x * n.x + mat[0].x * n.y + mat[0].x * n.z;
    out.y = mat[1].y * n.x + mat[1].y * n.y + mat[1].y * n.z;
    out.z = mat[2].z * n.x + mat[2].z * n.y + mat[2].z * n.z;
    out
}

pub fn transform_point_from_world_to_object_space(p: Vec3) -> Vec3 {
    if transform_list_size() == 0 {
        p
    } else {
        let mat = world_to_object_transform_matrix();
        transform_point(&mat, p)
    }
}

pub fn transform_vector_from_world_to_object_space(p: Vec3) -> Vec3 {
    if transform_list_size() == 0 {
        p
    } else {
        let mat = world_to_object_transform_matrix();
        transform_vector(&mat, p)
    }
}

pub fn transform_normal_from_world_to_object_space(p: Vec3) -> Vec3 {
    if transform_list_size() == 0 {
        p
    } else {
        let mat = world_to_object_transform_matrix();
        transform_normal(&mat, p)
    }
}

pub fn transform_point_from_object_to_world_space(p: Vec3) -> Vec3 {
    if transform_list_size() == 0 {
        p
    } else {
        let mat = object_to_world_transform_matrix();
        transform_point(&mat, p)
    }
}

pub fn transform_vector_from_object_to_world_space(p: Vec3) -> Vec3 {
    if transform_list_size() == 0 {
        p
    } else {
        let mat = object_to_world_transform_matrix();
        transform_vector(&mat, p)
    }
}

pub fn transform_normal_from_object_to_world_space(p: Vec3) -> Vec3 {
    if transform_list_size() == 0 {
        p
    } else {
        let mat = object_to_world_transform_matrix();
        transform_normal(&mat, p)
    }
}
