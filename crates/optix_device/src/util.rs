use crate::{intersect::get_attribute, payload::*};
use glam::Vec3;

pub fn pack_pointer<T>(ptr: *mut T) -> (u32, u32) {
    let x = ptr as u32;
    let y = (ptr as u64 >> 32) as u32;
    (x, y)
}

pub fn unpack_pointer<T>(x: u32, y: u32) -> *mut T {
    (((y as u64) << 32) | x as u64) as *mut T
}

pub fn store_vec3_payload(vec: Vec3, start_reg: u8) {
    set_payload(start_reg, vec.x.to_bits());
    set_payload(start_reg + 1, vec.y.to_bits());
    set_payload(start_reg + 2, vec.z.to_bits());
}

/// Retrieves a vector from the passed payload. This uses 3 payload
/// registers in total, starting from `start_reg`.
///
/// # Safety
///
/// `start_reg..(start_reg + 3)` payload slots must have all been set.
pub unsafe fn get_vec3_payload(start_reg: u8) -> Vec3 {
    let x = f32::from_bits(get_payload(start_reg));
    let y = f32::from_bits(get_payload(start_reg + 1));
    let z = f32::from_bits(get_payload(start_reg + 2));
    Vec3::new(x, y, z)
}

/// Retrieves a vector from the passed attributes. This uses 3 attribute
/// registers in total, starting from `start_reg`.
///
/// # Safety
///
/// `start_reg..(start_reg + 3)` attribute slots must have all been set.
pub unsafe fn get_vec3_attributes(start_reg: u8) -> Vec3 {
    let x = f32::from_bits(get_attribute(start_reg));
    let y = f32::from_bits(get_attribute(start_reg + 1));
    let z = f32::from_bits(get_attribute(start_reg + 2));
    Vec3::new(x, y, z)
}
