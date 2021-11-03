use crate::{acceleration::{BuildInput, TraversableHandle}, const_assert, const_assert_eq, sys};
use cust::{memory::DeviceSlice, DeviceCopy};
use cust_raw::CUdeviceptr;
use mint::RowMatrix3x4;

#[repr(C, align(16))]
#[derive(Debug, Copy, Clone, DeviceCopy)]
pub struct Instance {
    transform: RowMatrix3x4<f32>,
    instance_id: u32,
    sbt_offset: u32,
    visibility_mask: u32,
    flags: InstanceFlags,
    traversable_handle: TraversableHandle,
    pad: [u32; 2],
}

const_assert_eq!(std::mem::align_of::<Instance>(), sys::OptixInstanceByteAlignment);
const_assert_eq!(std::mem::size_of::<Instance>(), std::mem::size_of::<sys::OptixInstance>());


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
                0.0, 0.0, 1.0, 0.0].into(),
            instance_id: 0,
            sbt_offset: 0,
            visibility_mask: 255,
            flags: InstanceFlags::NONE,
            traversable_handle,
            pad: [0; 2],
        }
    }

    pub fn transform<T: Into<RowMatrix3x4<f32>>>(mut self, transform: T) -> Instance {
        self.transform = transform.into();
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
    instances: &'i DeviceSlice<Instance>,
}

impl<'i> InstanceArray<'i> {
    pub fn new(instances: &'i DeviceSlice<Instance>) -> InstanceArray {
        InstanceArray { instances }
    }
}

impl<'i> BuildInput for InstanceArray<'i> {
    fn to_sys(&self) -> sys::OptixBuildInput {
        cfg_if::cfg_if! {
            if #[cfg(any(feature="optix72", feature="optix73"))] {
                sys::OptixBuildInput {
                    type_: sys::OptixBuildInputType_OPTIX_BUILD_INPUT_TYPE_INSTANCES,
                    input: sys::OptixBuildInputUnion {
                        instance_array: std::mem::ManuallyDrop::new(sys::OptixBuildInputInstanceArray {
                            instances: self.instances.as_device_ptr(),
                            numInstances: self.instances.len() as u32,
                        })
                    }
                }
            } else {
                sys::OptixBuildInput {
                    type_: sys::OptixBuildInputType_OPTIX_BUILD_INPUT_TYPE_INSTANCES,
                    input: sys::OptixBuildInputUnion {
                        instance_array: std::mem::ManuallyDrop::new(sys::OptixBuildInputInstanceArray {
                            instances: self.instances.as_device_ptr(),
                            numInstances: self.instances.len() as u32,
                            aabbs: 0,
                            numAabbs: 0,
                        })
                    }
                }
            }
        }
    }
}

pub struct InstancePointerArray<'i> {
    instances: &'i DeviceSlice<CUdeviceptr>,
}

impl<'i> InstancePointerArray<'i> {
    pub fn new(instances: &'i DeviceSlice<CUdeviceptr>) -> InstancePointerArray {
        InstancePointerArray { instances }
    }
}

impl<'i> BuildInput for InstancePointerArray<'i> {
    fn to_sys(&self) -> sys::OptixBuildInput {
        cfg_if::cfg_if! {
            if #[cfg(any(feature="optix72", feature="optix73"))] {
                sys::OptixBuildInput {
                    type_: sys::OptixBuildInputType_OPTIX_BUILD_INPUT_TYPE_INSTANCE_POINTERS,
                    input: sys::OptixBuildInputUnion {
                        instance_array: std::mem::ManuallyDrop::new(sys::OptixBuildInputInstanceArray {
                            instances: self.instances.as_device_ptr(),
                            numInstances: self.instances.len() as u32,
                        })
                    }
                }
            } else {
                sys::OptixBuildInput {
                    type_: sys::OptixBuildInputType_OPTIX_BUILD_INPUT_TYPE_INSTANCE_POINTERS,
                    input: sys::OptixBuildInputUnion {
                        instance_array: std::mem::ManuallyDrop::new(sys::OptixBuildInputInstanceArray {
                            instances: self.instances.as_device_ptr(),
                            numInstances: self.instances.len() as u32,
                            aabbs: 0,
                            numAabbs: 0,
                        })
                    }
                }
            }
        }
    }
}

