#[cfg(target_os = "cuda")]
use core::arch::asm;
use cuda_std::gpu_only;
use glam::Vec3;
/// The type of primitive that a ray hit.
#[repr(u32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum HitKind {
    /// A custom primitive, the value is the custom
    /// 7-bit hit kind set when reporting an intersection.
    Custom(u8),
    /// B-spline curve of degree 2 with circular cross-section.
    RoundQuadraticBSpline,
    /// B-spline curve of degree 3 with circular cross-section.
    RoundCubicBSpline,
    /// Piecewise linear curve with circular cross-section.
    RoundLinear,
    /// CatmullRom curve with circular cross-section.
    RoundCatmullRom,
    /// ▲
    Triangle,
}

#[repr(u32)]
#[allow(dead_code)]
enum OptixPrimitiveType {
    Custom = 0x2500,
    RoundQuadraticBSpline = 0x2501,
    RoundCubicBSpline = 0x2502,
    RoundLinear = 0x2503,
    RoundCatmullRom = 0x2504,
    Triangle = 0x2531,
}

#[gpu_only]
fn get_primitive_type(val: u8) -> OptixPrimitiveType {
    let raw: u32;
    unsafe {
        asm!("call ({}), _optix_get_primitive_type_from_hit_kind, ({});", out(reg32) raw, in(reg32) val);
        core::mem::transmute(raw)
    }
}

impl HitKind {
    pub(crate) fn from_raw(val: u8) -> Self {
        let kind = get_primitive_type(val);
        match kind {
            OptixPrimitiveType::Custom => HitKind::Custom(val & 0b0111_1111),
            OptixPrimitiveType::RoundQuadraticBSpline => HitKind::RoundQuadraticBSpline,
            OptixPrimitiveType::RoundCubicBSpline => HitKind::RoundCubicBSpline,
            OptixPrimitiveType::RoundLinear => HitKind::RoundLinear,
            OptixPrimitiveType::RoundCatmullRom => HitKind::RoundCatmullRom,
            OptixPrimitiveType::Triangle => HitKind::Triangle,
        }
    }
}

#[gpu_only]
fn get_hit_kind() -> u8 {
    let x: u8;
    unsafe {
        asm!("call ({}), _optix_get_hit_kind, ();", out(reg32) x);
    }
    x
}

/// Returns the kind of primitive that was hit by the ray.
pub fn hit_kind() -> HitKind {
    HitKind::from_raw(get_hit_kind())
}

#[gpu_only]
pub fn is_back_face_hit() -> bool {
    let hit_kind = get_hit_kind();
    let x: u32;
    unsafe {
        asm!("call ({}), _optix_get_backface_from_hit_kind, ({});", out(reg32) x, in(reg32) hit_kind);
    }
    x == 1
}

/// Whether the ray hit a front face.
pub fn is_front_face_hit() -> bool {
    !is_back_face_hit()
}

/// Whether the ray hit a triangle.
pub fn is_triangle_hit() -> bool {
    is_triangle_front_face_hit() || is_triangle_back_face_hit()
}

const OPTIX_HIT_KIND_TRIANGLE_FRONT_FACE: u8 = 0xFE;
const OPTIX_HIT_KIND_TRIANGLE_BACK_FACE: u8 = 0xFF;

/// Whether the ray hit the front face of a triangle.
pub fn is_triangle_front_face_hit() -> bool {
    get_hit_kind() == OPTIX_HIT_KIND_TRIANGLE_FRONT_FACE
}

/// Whether the ray hit the back face of a triangle.
pub fn is_triangle_back_face_hit() -> bool {
    get_hit_kind() == OPTIX_HIT_KIND_TRIANGLE_BACK_FACE
}

/// Returns the barycentric coordinates of the hit point on the hit triangle.
#[gpu_only]
pub fn triangle_barycentrics() -> Vec3 {
    let x: f32;
    let y: f32;
    unsafe {
        asm!(
            "call ({}, {}), _optix_get_triangle_barycentrics, ();",
            out(reg32) x,
            out(reg32) y
        );
    }
    Vec3::new(x, y, 1.0 - x - y)
}
