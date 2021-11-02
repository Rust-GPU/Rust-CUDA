use crate::{
    acceleration::{GeometryFlags, TraversableHandle}, context::DeviceContext, error::Error, module::Module, optix_call,
    sys,
};
use cust::{memory::DSlice, DeviceCopy};
use std::ffi::c_void;

#[repr(C, align(16))]
#[derive(Debug, DeviceCopy, Copy, Clone)]
pub struct Instance {
    transform: [f32; 12],
    instance_id: u32,
    sbt_offset: u32,
    visibility_mask: u32,
    flags: InstanceFlags,
    traversable_handle: TraversableHandle,
    pad: [u32; 2],
}


bitflags::bitflags! {
    #[derive(DeviceCopy)]
    pub struct InstanceFlags: u32 {
        const NONE = sys::OptixInstanceFlags_OPTIX_INSTANCE_FLAG_NONE;
        const DISABLE_TRIANGLE_FACE_CULLING = sys::OptixInstanceFlags_OPTIX_INSTANCE_FLAG_DISABLE_TRIANGLE_FACE_CULLING;
        const FLIP_TRIANGLE_FACING = sys::OptixInstanceFlags_OPTIX_INSTANCE_FLAG_FLIP_TRIANGLE_FACING;
        const DISABLE_ANYHIT = sys::OptixInstanceFlags_OPTIX_INSTANCE_FLAG_DISABLE_ANYHIT;
        const ENFORCE_ANYHIT = sys::OptixInstanceFlags_OPTIX_INSTANCE_FLAG_ENFORCE_ANYHIT;
        const DISABLE_TRANSFORM = sys::OptixInstanceFlags_OPTIX_INSTANCE_FLAG_DISABLE_TRANSFORM;
    }
}

impl Instance {
    pub fn new(traversable_handle: TraversableHandle) -> Instance {
        #[cfg_attr(rustfmt, rustfmt_skip)]
        Instance {
            transform: [
                1.0, 0.0, 0.0, 0.0, 
                0.0, 1.0, 0.0, 0.0, 
                0.0, 0.0, 1.0, 0.0],
            instance_id: 0,
            sbt_offset: 0,
            visibility_mask: 255,
            flags: InstanceFlags::NONE,
            traversable_handle,
            pad: [0; 2],
        }
    }

    pub fn transform(mut self, transform: [f32; 12]) -> Instance {
        self.transform = transform;
        self
    }

    pub fn instance_id(mut self, instance_id: u32) -> Instance {
        self.instance_id = instance_id;
        self
    }

    pub fn sbt_offset(mut self, sbt_offset: u32) -> Instance {
        self.sbt_offset = sbt_offset;
        self
    }

    pub fn visibility_mask(mut self, visibility_mask: u8) -> Instance {
        self.visibility_mask = visibility_mask as u32;
        self
    }

    pub fn flags(mut self, flags: InstanceFlags) -> Instance {
        self.flags = flags;
        self
    }
}

pub struct InstanceArray<'i> {
    instances: &'i DSlice<Instance>,
}

impl<'i> InstanceArray<'i> {
    pub fn new(instances: &'i DSlice<Instance>) -> InstanceArray {
        InstanceArray {
            instances
        }
    }
}

pub trait BuildInputInstanceArray {
    fn to_sys(&self) -> sys::OptixBuildInputInstanceArray;
}

impl BuildInputInstanceArray for () {
    fn to_sys(&self) -> sys::OptixBuildInputInstanceArray {
        unreachable!()
    }
}

impl<'i> BuildInputInstanceArray for InstanceArray<'i> {
    fn to_sys(&self) -> sys::OptixBuildInputInstanceArray {
        cfg_if::cfg_if! {
            if #[cfg(any(feature="optix72", feature="optix73"))] {
                sys::OptixBuildInputInstanceArray {
                    instances: self.instances.as_device_ptr().as_raw() as u64,
                    numInstances: self.instances.len() as u32,
                }
            } else {
                sys::OptixBuildInputInstanceArray {
                    instances: self.instances.as_device_ptr().as_raw() as u64,
                    numInstances: self.instances.len() as u32,
                    aabbs: 0,
                    numAabbs: 0,
                }
            }
        }
    }
}

