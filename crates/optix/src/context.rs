//! OptiX Device Context handling.

use std::{ffi::c_void, mem::MaybeUninit, ptr};

use cust::context::ContextHandle;

use crate::{error::OptixResult, optix_call, sys};

/// A certain property belonging to an OptiX device.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OptixDeviceProperty {
    /// The maximum value that can be given to the OptiX pipeline's max trace depth.
    MaxTraceDepth,
    /// The maximum value that can be given to the OptiX pipeline's stack size method's max traversable
    /// graph depth.
    MaxTraversableGraphDepth,
    /// The maximum number of primitives allowed (over all build inputs) in a single Geometry Acceleration Structure (GAS).
    MaxPrimitivesPerGas,
    /// The maximum number of instances allowed (over all build inputs) in a single Instance Accceleration Structure (IAS).
    MaxInstancesPerIas,
    /// The RT core version supported by the device (0 for no support, 10 for version 1.0).
    RtCoreVersion,
    /// The maximum value for an OptiX instance's ID.
    MaxInstanceId,
    /// The number of bits available for an OptiX instance's visibility mask. Bits higher than that must be set to zero.
    NumBitsInstanceVisibilityMask,
    /// The maximum number of instances that can be added to a single Instance Acceleration Structure (IAS).
    MaxSbtRecordsPerGas,
    /// The maximum value for an OptiX instance's sbt offset.
    MaxSbtOffset,
}

impl OptixDeviceProperty {
    // we could repr this the same as the sys version, but for better compatability
    // and safety in the future, we just match.
    pub fn to_raw(self) -> sys::OptixDeviceProperty {
        use OptixDeviceProperty::*;
        match self {
        MaxTraceDepth => sys::OptixDeviceProperty::OPTIX_DEVICE_PROPERTY_LIMIT_MAX_TRACE_DEPTH,
        MaxTraversableGraphDepth => sys::OptixDeviceProperty::OPTIX_DEVICE_PROPERTY_LIMIT_MAX_TRAVERSABLE_GRAPH_DEPTH,
        MaxPrimitivesPerGas => sys::OptixDeviceProperty::OPTIX_DEVICE_PROPERTY_LIMIT_MAX_PRIMITIVES_PER_GAS,
        MaxInstancesPerIas => sys::OptixDeviceProperty::OPTIX_DEVICE_PROPERTY_LIMIT_MAX_INSTANCES_PER_IAS,
        RtCoreVersion => sys::OptixDeviceProperty::OPTIX_DEVICE_PROPERTY_RTCORE_VERSION,
        MaxInstanceId => sys::OptixDeviceProperty::OPTIX_DEVICE_PROPERTY_LIMIT_MAX_INSTANCE_ID,
        NumBitsInstanceVisibilityMask => sys::OptixDeviceProperty::OPTIX_DEVICE_PROPERTY_LIMIT_NUM_BITS_INSTANCE_VISIBILITY_MASK,
        MaxSbtRecordsPerGas => sys::OptixDeviceProperty::OPTIX_DEVICE_PROPERTY_LIMIT_MAX_SBT_RECORDS_PER_GAS,
        MaxSbtOffset => sys::OptixDeviceProperty::OPTIX_DEVICE_PROPERTY_LIMIT_MAX_SBT_OFFSET,
        }
    }
}

#[derive(Debug)]
pub struct OptixContext {
    pub(crate) raw: sys::OptixDeviceContext,
}

impl Drop for OptixContext {
    fn drop(&mut self) {
        unsafe {
            sys::optixDeviceContextDestroy(self.raw);
        }
    }
}

impl OptixContext {
    // TODO(RDambrosio016): expose device context options

    /// Creates a new [`OptixContext`] from a cust CUDA context.
    pub fn new(cuda_ctx: &impl ContextHandle) -> OptixResult<Self> {
        let mut raw = MaybeUninit::uninit();
        unsafe {
            optix_call!(optixDeviceContextCreate(
                cuda_ctx.get_inner(),
                ptr::null(),
                raw.as_mut_ptr()
            ))?;
            Ok(Self {
                raw: raw.assume_init(),
            })
        }
    }

    pub fn get_property(&self, property: OptixDeviceProperty) -> OptixResult<u32> {
        let raw_prop = property.to_raw();
        unsafe {
            let mut value = 0u32;
            optix_call!(optixDeviceContextGetProperty(
                self.raw,
                raw_prop,
                &mut value as *mut _ as *mut c_void,
                4,
            ))?;
            Ok(value)
        }
    }

    pub fn as_raw(&self) -> sys::OptixDeviceContext {
        self.raw
    }
}
