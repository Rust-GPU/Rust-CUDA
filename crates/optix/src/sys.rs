#![allow(warnings)]

use cust_raw::*;

use std::mem::ManuallyDrop;

type size_t = usize;

include!(concat!(env!("OUT_DIR"), "/optix_wrapper.rs"));

extern "C" {
    pub fn optixInit() -> OptixResult;
}

// The SBT record header is an opaque blob used by optix
#[repr(C)]
#[derive(Default, Clone, Copy)]
pub struct SbtRecordHeader {
    header: [u8; OptixSbtRecordHeaderSize as usize],
}

impl SbtRecordHeader {
    pub fn as_mut_ptr(&mut self) -> *mut std::os::raw::c_void {
        self.header.as_mut_ptr() as *mut std::os::raw::c_void
    }
}

// Manually define the build input union as the bindgen is pretty nasty
#[repr(C)]
pub union OptixBuildInputUnion {
    pub triangle_array: ManuallyDrop<OptixBuildInputTriangleArray>,
    pub curve_array: ManuallyDrop<OptixBuildInputCurveArray>,
    pub custom_primitive_array: ManuallyDrop<OptixBuildInputCustomPrimitiveArray>,
    pub instance_array: ManuallyDrop<OptixBuildInputInstanceArray>,
    pad: [std::os::raw::c_char; 1024],
}

impl Default for OptixBuildInputUnion {
    fn default() -> OptixBuildInputUnion {
        OptixBuildInputUnion { pad: [0i8; 1024] }
    }
}

#[repr(C)]
pub struct OptixBuildInput {
    pub type_: OptixBuildInputType,
    pub input: OptixBuildInputUnion,
}

// Sanity check that the size of this union we're defining matches the one in
// optix header so we don't get any nasty surprises
fn _size_check() {
    unsafe {
        std::mem::transmute::<OptixBuildInput, [u8; OptixBuildInputSize]>(OptixBuildInput {
            type_: OptixBuildInputType_OPTIX_BUILD_INPUT_TYPE_TRIANGLES,
            input: { OptixBuildInputUnion { pad: [0; 1024] } },
        });
    }
}
