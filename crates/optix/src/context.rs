use std::os::raw::{c_char, c_uint};
use std::{
    ffi::{c_void, CStr, CString},
    mem::MaybeUninit,
};

use cust::context::ContextHandle;

use crate::{error::Error, optix_call, sys};
type Result<T, E = Error> = std::result::Result<T, E>;

/// A certain property belonging to an OptiX device.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeviceProperty {
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

impl DeviceProperty {
    // we could repr this the same as the sys version, but for better compatability
    // and safety in the future, we just match.
    pub fn to_raw(self) -> sys::OptixDeviceProperty::Type {
        use DeviceProperty::*;
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

    /// Creates a new [`DeviceContext`] from a cust CUDA context.
    ///
    /// If `enable_validation` is `true`, then additional tests and settings are
    /// enabled during application execution. This additional processing can reduce
    /// performance, so it should only be used during debugging or in the final
    /// testing phase of a completed application.
    pub fn new(cuda_ctx: &impl ContextHandle, enable_validation: bool) -> Result<Self> {
        let mut raw = MaybeUninit::uninit();

        let mut opt = sys::OptixDeviceContextOptions::default();
        if enable_validation {
            opt.validationMode =
                sys::OptixDeviceContextValidationMode_OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL;
        }

        unsafe {
            optix_call!(optixDeviceContextCreate(
                cuda_ctx.get_inner(),
                &opt,
                raw.as_mut_ptr()
            ))?;
            Ok(Self {
                raw: raw.assume_init(),
            })
        }
    }

    /// Returns the low and high water marks, respectively, for disk cache garbage collection.
    /// If the cache has been disabled by setting the environment variable
    /// OPTIX_CACHE_MAXSIZE=0, this function will return 0 for the low and high water marks.
    pub fn get_cache_database_sizes(&self) -> Result<(usize, usize)> {
        let mut low = 0;
        let mut high = 0;
        unsafe {
            Ok(optix_call!(optixDeviceContextGetCacheDatabaseSizes(
                self.raw, &mut low, &mut high,
            ))
            .map(|_| (low, high))?)
        }
    }

    /// Indicated whether the disk cache is enabled
    pub fn get_cache_enabled(&self) -> Result<bool> {
        let mut result = 0;
        unsafe {
            Ok(
                optix_call!(optixDeviceContextGetCacheEnabled(self.raw, &mut result,))
                    .map(|_| result != 0)?,
            )
        }
    }

    /// Returns the location of the disk cache. If the cache has been disabled
    /// by setting the environment variable OPTIX_CACHE_MAXSIZE=0, this function will return an empy string.
    pub fn get_cache_location(&self) -> Result<String> {
        let mut buf = [0i8; 1024];
        unsafe {
            Ok(optix_call!(optixDeviceContextGetCacheLocation(
                self.raw,
                buf.as_mut_ptr(),
                buf.len(),
            ))
            .map(|_| CStr::from_ptr(buf.as_ptr()).to_string_lossy().to_string())?)
        }
    }

    /// Query properties of this context.
    pub fn get_property(&self, property: DeviceProperty) -> Result<u32> {
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

    /// Sets the low and high water marks for disk cache garbage collection.
    ///
    /// Garbage collection is triggered when a new entry is written to the cache
    /// and the current cache data size plus the size of the cache entry that is
    /// about to be inserted exceeds the high water mark. Garbage collection proceeds
    /// until the size reaches the low water mark. Garbage collection will always
    /// free enough space to insert the new entry without exceeding the low water
    /// mark. Setting either limit to zero will disable garbage collection. An
    /// error will be returned if both limits are non-zero and the high water mark
    /// is smaller than the low water mark.
    ///
    /// Note that garbage collection is performed only on writes to the disk cache.
    /// No garbage collection is triggered on disk cache initialization or immediately
    /// when calling this function, but on subsequent inserting of data into the
    /// database.
    ///
    /// If the size of a compiled module exceeds the value configured for the high
    /// water mark and garbage collection is enabled, the module will not be added
    /// to the cache and a warning will be added to the log.
    ///
    /// The high water mark can be overridden with the environment variable
    /// OPTIX_CACHE_MAXSIZE. The environment variable takes precedence over the
    /// function parameters. The low water mark will be set to half the value of
    /// OPTIX_CACHE_MAXSIZE. Setting OPTIX_CACHE_MAXSIZE to 0 will disable the
    /// disk cache, but will not alter the contents of the cache. Negative and
    /// non-integer values will be ignored.    
    pub fn set_cache_database_sizes(&mut self, low: usize, high: usize) -> Result<()> {
        unsafe {
            Ok(optix_call!(optixDeviceContextSetCacheDatabaseSizes(
                self.raw, low, high,
            ))?)
        }
    }

    /// Enables or disables the disk cache.
    ///
    /// If caching was previously disabled, enabling it will attempt to initialize
    /// the disk cache database using the currently configured cache location.
    /// An error will be returned if initialization fails.
    ///
    /// Note that no in-memory cache is used, so no caching behavior will be observed
    /// if the disk cache is disabled.
    ///
    /// The cache can be disabled by setting the environment variable
    /// OPTIX_CACHE_MAXSIZE=0. The environment variable takes precedence over this
    /// setting. See optixDeviceContextSetCacheDatabaseSizes for additional information.
    ///
    /// Note that the disk cache can be disabled by the environment variable, but
    /// it cannot be enabled via the environment if it is disabled via the API.    
    pub fn set_cache_enabled(&mut self, enable: bool) -> Result<()> {
        unsafe {
            Ok(optix_call!(optixDeviceContextSetCacheEnabled(
                self.raw,
                if enable { 1 } else { 0 }
            ))?)
        }
    }

    /// Sets the location of the disk cache.
    ///
    /// The location is specified by a directory. This directory should not be used for other purposes and will be created if it does not exist. An error will be returned if is not possible to create the disk cache at the specified location for any reason (e.g., the path is invalid or the directory is not writable). Caching will be disabled if the disk cache cannot be initialized in the new location. If caching is disabled, no error will be returned until caching is enabled. If the disk cache is located on a network file share, behavior is undefined.
    ///
    /// The location of the disk cache can be overridden with the environment variable OPTIX_CACHE_PATH. The environment variable takes precedence over this setting.
    ///
    /// The default location depends on the operating system:
    ///
    /// * Windows: `LOCALAPPDATA%\NVIDIA\OptixCache`
    /// * Linux: `/var/tmp/OptixCache_<username>` (or `/tmp/OptixCache_<username>`
    ///     if the first choice is not usable), the underscore and username suffix are omitted if the username cannot be obtained
    /// * MacOS X:  `/Library/Application Support/NVIDIA/OptixCache`
    pub fn set_cache_location(&mut self, location: &str) -> Result<()> {
        let location = CString::new(location).map_err(|_| Error::NulBytesInString)?;
        unsafe {
            Ok(optix_call!(optixDeviceContextSetCacheLocation(
                self.raw,
                location.as_ptr()
            ))?)
        }
    }

    /// Sets the current log callback method.
    ///
    /// The following log levels are defined:
    /// * 0 - disable: Setting the callback level will disable all messages. The
    ///   callback function will not be called in this case.
    /// * 1 - fatal: A non-recoverable error. The context and/or OptiX itself might no
    ///   longer be in a usable state.
    /// * 2 - error: A recoverable error, e.g., when passing invalid call parameters.
    /// * 3 - warning: Hints that OptiX might not behave exactly as requested by the
    ///   user or may perform slower than expected.
    /// * 4 - print: Status or progress messages. Higher levels might occur.
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

    /// Get the FFI context representation
    pub fn as_raw(&self) -> sys::OptixDeviceContext {
        self.raw
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
