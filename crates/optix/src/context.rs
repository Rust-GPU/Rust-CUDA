//! OptiX Device Context handling.

use std::os::raw::{c_char, c_uint};
use std::{
    ffi::{c_void, CStr},
    mem::MaybeUninit,
    ptr,
};

use cust::context::ContextHandle;

use crate::{error::Error, optix_call, sys};
type Result<T, E = Error> = std::result::Result<T, E>;

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
    pub fn to_raw(self) -> sys::OptixDeviceProperty::Type {
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
#[repr(transparent)]
pub struct DeviceContext {
    pub(crate) raw: sys::OptixDeviceContext,
}

impl Drop for DeviceContext {
    fn drop(&mut self) {
        unsafe {
            sys::optixDeviceContextDestroy(self.raw);
        }
    }
}

impl DeviceContext {
    // TODO(RDambrosio016): expose device context options

    /// Creates a new [`OptixContext`] from a cust CUDA context.
    pub fn new(cuda_ctx: &impl ContextHandle) -> Result<Self> {
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

    pub fn get_property(&self, property: OptixDeviceProperty) -> Result<u32> {
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

    /// Sets the current log callback method.
    ///
    /// The following log levels are defined.
    /// * 0 - disable: Setting the callback level will disable all messages. The
    /// callback function will not be called in this case.
    /// * 1 - fatal: A non-recoverable error. The context and/or OptiX itself
    ///   might
    /// no longer be in a usable state.
    /// * 2 - error: A recoverable error, e.g., when passing invalid call
    /// parameters.
    /// * 3 - warning: Hints that OptiX might not behave exactly as requested by
    /// the user or may perform slower than expected.
    /// * 4 - print: Status or progress messages.
    /// Higher levels might occur.
    pub fn set_log_callback<F>(&mut self, cb: F, level: u32) -> Result<()>
    where
        F: FnMut(u32, &str, &str) + 'static,
    {
        let (closure, trampoline) = unsafe { unpack_closure(cb) };
        unsafe {
            Ok(optix_call!(optixDeviceContextSetLogCallback(
                self.raw,
                Some(trampoline),
                closure,
                level
            ))?)
        }
    }
}

type LogCallback = extern "C" fn(c_uint, *const c_char, *const c_char, *mut c_void);

/// Unpack a Rust closure, extracting a `void*` pointer to the data and a
/// trampoline function which can be used to invoke it.
///
/// # Safety
///
/// It is the user's responsibility to ensure the closure outlives the returned
/// `void*` pointer.
///
/// Calling the trampoline function with anything except the `void*` pointer
/// will result in *Undefined Behaviour*.
unsafe fn unpack_closure<F>(closure: F) -> (*mut c_void, LogCallback)
where
    F: FnMut(u32, &str, &str),
{
    extern "C" fn trampoline<F>(
        level: c_uint,
        tag: *const c_char,
        msg: *const c_char,
        data: *mut c_void,
    ) where
        F: FnMut(u32, &str, &str),
    {
        if let Err(e) = std::panic::catch_unwind(|| {
            let tag = unsafe { CStr::from_ptr(tag).to_string_lossy().into_owned() };
            let msg = unsafe { CStr::from_ptr(msg).to_string_lossy().into_owned() };
            let closure: &mut F = unsafe { &mut *(data as *mut F) };

            (*closure)(level, &tag, &msg);
        }) {
            eprintln!("Caught a panic calling log closure: {:?}", e);
        }
    }

    let cb = Box::new(closure);
    let cb = Box::leak(cb);

    (cb as *mut F as *mut c_void, trampoline::<F>)
}
