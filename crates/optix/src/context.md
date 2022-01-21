# OptiX Device Context handling.

# Programming Guide...
<details>
<summary>Click here to expand programming guide</summary>

A context is created by [`DeviceContext::new()`] and is used to manage a single
GPU. The NVIDIA OptiX 7 device context is created by specifying the CUDA
context associated with the device.

```
# fn doit() -> Result<(), Box<dyn std::error::Error>> {
use optix::prelude as ox;
use cust::prelude as cu;

// Initialize cuda and optix
cust::init(cu::CudaFlags::empty())?;
ox::init()?;

// Create a cuda context for the first device
let device = cu::Device::get_device(0)?;
let cu_ctx = cu::Context::create_and_push(cu::ContextFlags::SCHED_AUTO |
cu::ContextFlags::MAP_HOST, device)?;

// Create optix device context
let ctx = ox::DeviceContext::new(&cu_ctx, false)?;

# Ok(())
# }
```
A small set of context properties exist for determining sizes and limits. These
are queried using [`DeviceContext::get_property()`]. Such properties include
maximum trace depth, maximum traversable graph depth, maximum primitives per
build input, and maximum number of instances per acceleration structure.

The context may retain ownership of any GPU resources necessary to launch the
ray tracing kernels. Some API objects will retain host memory. These are defined
with create/destroy patterns in the API. The context's `Drop` impl will clean
up any host or device resources associated with the context. If any other API
objects associated with this context still exist when the context is destroyed,
they are also destroyed.

An application may combine any mixture of supported GPUs as long as the data
transfer and synchronization is handled appropriately. Some applications may
choose to simplify multi-GPU handling by restricting the variety of these blends,
for example, by mixing only GPUs of the same streaming multiprocessor version
to simplify data sharing.

## Logging callbacks

A logging callback closure can be specified using [`DeviceContext::set_log_callback`].
The closure has the signiature:
`F: FnMut(u32, &str, &str) + 'static`

The first argument is the log level and indicates the serverity of the message:
 
* 0 - disable: Setting the callback level will disable all messages. The
callback function will not be called in this case.
* 1 - fatal: A non-recoverable error. The context and/or OptiX itself
  might
no longer be in a usable state.
* 2 - error: A recoverable error, e.g., when passing invalid call
parameters.
* 3 - warning: Hints that OptiX might not behave exactly as requested by
the user or may perform slower than expected.
* 4 - print: Status or progress messages.
Higher levels might occur.
The second argument is a message category description (for example, "SCENE STAT")
The last argument is the message itself.

## Compilation caching

Compilation of input programs will be cached to disk when creating [`Module`](crate::module::Module),
[`ProgramGroup`](crate::program_group::ProgramGroup), and
[`Pipeline`](crate::pipeline::Pipeline) objects if caching has been enabled.

Subsequent compilation can reuse the cached data to improve the time to create
these objects. The cache can be shared between multiple [`DeviceContext`]
objects, and NVIDIA OptiX 7 will take care of ensuring correct multi-threaded
access to the cache. If no sharing between [`DeviceContext`] objects is desired,
the path to the cache can be set differently for each [`DeviceContext`].
Caching can be disabled entirely by setting the environment variable
`OPTIX_CACHE_MAXSIZE` to 0. Disabling the cache via the environment variable
will not affect existing cache files or their contents.

The disk cache can be controlled with:

### [`DeviceContext::set_cache_enabled()`]
The cache database is initialized when the device context is created and when
enabled through this function call. If the database cannot be initialized when
the device context is created, caching will be disabled; a message is reported
to the log callback if caching is enabled. In this case, the call to
[`DeviceContext::new()`] does not return an error. To ensure that cache
initialization succeeded on context creation, the status can be queried using
[`DeviceContext::get_cache_enabled`]. If caching is disabled, the cache can be
reconfigured and then enabled using [`DeviceContext::set_cache_enabled`]. If
the cache database cannot be initialized, an error is returned. Garbage
collection is performed on the next write to the cache database, not when the
cache is enabled.

### [`DeviceContext::set_cache_location`]
The disk cache is created in the directory specified by location. The directory
is created if it does not exist.

The cache database is created immediately if the cache is currently enabled.
Otherwise the cache database is created later when the cache is enabled. An
error is returned if it is not possible to create the cache database file at
the specified location for any reason (for example, if the path is invalid or
if the directory is not writable) and caching will be disabled. If the disk
cache is located on a network file share, behavior is undefined.

The location of the disk cache can be overridden with the environment variable
`OPTIX_CACHE_PATH`. This environment variable takes precedence over the value
passed to this function when the disk cache is enabled.

The default location of the cache depends on the operating system:
* Windows -	`%LOCALAPPDATA%\NVIDIA\OptixCache`
* Linux	- `/var/tmp/OptixCache_username`, or `/tmp/OptixCache_username` if the
first choice is not usable. The underscore and username suffix are omitted if
the username cannot be obtained.

### [`DeviceContext::set_cache_database_sizes()`]
Parameters `low` and `high` set the low and high water marks for disk cache
garbage collection. Setting either limit to zero disables garbage collection.
Garbage collection only happens when the cache database is written. It is
triggered whenever the cache data size exceeds the high water mark and proceeding
until the size reaches the low water mark. Garbage collection always frees enough
space to allow the insertion of the new entry within the boundary of the low
water mark. An error is returned if either limit is nonzero and the high water
mark is lower than the low water mark. If more than one device context accesses
the same cache database with different high and low water mark values, the device
context uses its values when writing to the cache database.

The high water mark can be overridden with the environment variable
`OPTIX_CACHE_MAXSIZE`. Setting `OPTIX_CACHE_MAXSIZE` to 0 will disable the cache.
Negative and non-integer values will be ignored.

`OPTIX_CACHE_MAXSIZE` takes precedence over the `high` value passed to this
function. The low water mark will be set to half the value of
`OPTIX_CACHE_MAXSIZE`.

Corresponding `get_xxx()` functions are supplied to retrieve the current value of these
cache properties.

## Validation Mode
The NVIDIA OptiX 7 validation mode can help uncover errors which might otherwise
go undetected or which occur only intermittently and are difficult to locate.
Validation mode enables additional tests and settings during application
execution. This additional processing can reduce performance, so it should only
be used during debugging or in the final testing phase of a completed application.

Validation mode can be enabled by passing `true` to the `enable_validation`
parameter of [`DeviceContext::new()`].

[`OptixError::ValidationFailure`](crate::error::OptixError::ValidationFailure)
will be signalled if an error is caught when validation mode is enabled.
[`launch()`](crate::launch) will synchronize after the launch and report errors,
if any.

Among other effects, validation mode implicitly enables all OptiX debug
exceptions and provides an exception program if none is provided. The first
non-user exception caught inside an exception program will therefore be reported
and the launch terminated immediately. This will make exceptions more visible
that otherwise might be overlooked.

</details>

