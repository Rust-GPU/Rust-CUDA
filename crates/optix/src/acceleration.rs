#![allow(clippy::missing_safety_doc)]

use crate::{const_assert, const_assert_eq, context::DeviceContext, error::Error, optix_call};
use cust::memory::{
    CopyDestination, DeviceBox, DeviceBuffer, DeviceCopy, DevicePointer, DeviceSlice,
};
type Result<T, E = Error> = std::result::Result<T, E>;

use memoffset::offset_of;
use std::ffi::c_void;
use std::mem::size_of;
use std::ops::Deref;
use std::{
    collections::hash_map::DefaultHasher,
    hash::{Hash, Hasher},
    marker::PhantomData,
};

use cust_raw::driver_sys::CUdeviceptr;
use mint::{RowMatrix3x4, Vector3};

pub trait BuildInput: std::hash::Hash {
    fn to_sys(&self) -> optix_sys::OptixBuildInput;
}

pub trait Traversable {
    fn handle(&self) -> TraversableHandle;
}

/// Wrapper struct containing the storage and handle for a static acceleration
/// structure.
///
/// An Accel can be built by providing a slice of [`BuildInput`]s over which to
/// build the acceleration structure, together with a matching slice of
/// [`AccelBuildOptions`].
///
/// ```no_run
/// use cust::prelude as cu;
/// use optix::prelude as ox;
/// # fn doit() -> Result<(), Box<dyn std::error::Error>> {
/// # cust::init(cu::CudaFlags::empty())?;
/// # ox::init()?;
/// # let device = cu::Device::get_device(0)?;
/// # let cu_ctx = cu::Context::create_and_push(cu::ContextFlags::SCHED_AUTO |
/// # cu::ContextFlags::MAP_HOST, device)?;
/// # let ctx = ox::DeviceContext::new(&cu_ctx, false)?;
/// # let vertices: Vec<[f32; 3]> = Vec::new();
/// # let indices: Vec<[u32; 3]> = Vec::new();
/// # let stream = cu::Stream::new(cu::StreamFlags::DEFAULT, None)?;
///
/// let buf_vertex = cu::DeviceBuffer::from_slice(&vertices)?;
/// let buf_indices = cu::DeviceBuffer::from_slice(&indices)?;
///
/// let geometry_flags = ox::GeometryFlags::None;
/// let triangle_input =
///     ox::IndexedTriangleArray::new(
///         &[&buf_vertex],
///         &buf_indices,
///         &[geometry_flags]
///     );
///
/// let accel_options =
///     ox::AccelBuildOptions::new(
///         ox::BuildFlags::ALLOW_COMPACTION,
///         ox::BuildOperation::Build
///     );
///
/// let build_inputs = vec![triangle_input];
///
/// let gas = ox::Accel::build(
///     &ctx,
///     &stream,
///     &[accel_options],
///     &build_inputs,
///     true
/// )?;
///
/// stream.synchronize()?;
/// # Ok(())
/// # }
/// ```
pub struct Accel {
    #[allow(dead_code)]
    buf: DeviceBuffer<u8>,
    hnd: TraversableHandle,
}

impl Traversable for Accel {
    /// Get the [`TraversableHandle`] that represents this accel.
    fn handle(&self) -> TraversableHandle {
        self.hnd
    }
}

impl Accel {
    /// Build and (optionally) compact the acceleration structure for the given
    /// `build_inputs`.
    ///
    /// This will handle all necessary memory allocation internally, synchronizing
    /// all internal steps, but NOT the final build or compaction.
    ///
    /// If you want to re-use buffers between builds and line up multiple builds
    /// at once for more performance/efficiency, you should use the unsafe api.
    ///
    /// ```no_run
    /// use cust::prelude as cu;
    /// use optix::prelude as ox;
    /// # fn doit() -> Result<(), Box<dyn std::error::Error>> {
    /// # cust::init(cu::CudaFlags::empty())?;
    /// # ox::init()?;
    /// # let device = cu::Device::get_device(0)?;
    /// # let cu_ctx = cu::Context::create_and_push(cu::ContextFlags::SCHED_AUTO |
    /// # cu::ContextFlags::MAP_HOST, device)?;
    /// # let ctx = ox::DeviceContext::new(&cu_ctx, false)?;
    /// # let vertices: Vec<[f32; 3]> = Vec::new();
    /// # let indices: Vec<[u32; 3]> = Vec::new();
    /// # let stream = cu::Stream::new(cu::StreamFlags::DEFAULT, None)?;
    ///
    /// let buf_vertex = cu::DeviceBuffer::from_slice(&vertices)?;
    /// let buf_indices = cu::DeviceBuffer::from_slice(&indices)?;
    ///
    /// let geometry_flags = ox::GeometryFlags::None;
    /// let triangle_input =
    ///     ox::IndexedTriangleArray::new(
    ///         &[&buf_vertex],
    ///         &buf_indices,
    ///         &[geometry_flags]
    ///     );
    ///
    /// let accel_options =
    ///     ox::AccelBuildOptions::new(
    ///         ox::BuildFlags::ALLOW_COMPACTION,
    ///         ox::BuildOperation::Build
    ///     );
    ///
    /// let build_inputs = vec![triangle_input];
    ///
    /// let gas = ox::Accel::build(
    ///     &ctx,
    ///     &stream,
    ///     &[accel_options],
    ///     &build_inputs,
    ///     true
    /// )?;
    ///
    /// stream.synchronize()?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn build<I: BuildInput>(
        ctx: &DeviceContext,
        stream: &cust::stream::Stream,
        accel_options: &[AccelBuildOptions],
        build_inputs: &[I],
        compact: bool,
    ) -> Result<Accel> {
        let sizes = accel_compute_memory_usage(ctx, accel_options, build_inputs)?;
        let mut output_buffer =
            unsafe { DeviceBuffer::<u8>::uninitialized(sizes.output_size_in_bytes)? };

        let mut temp_buffer =
            unsafe { DeviceBuffer::<u8>::uninitialized(sizes.temp_size_in_bytes)? };

        let compacted_size_buffer = unsafe { DeviceBox::<usize>::uninitialized()? };

        let mut properties = vec![AccelEmitDesc::CompactedSize(
            compacted_size_buffer.as_device_ptr(),
        )];

        let hnd = unsafe {
            accel_build(
                ctx,
                stream,
                accel_options,
                build_inputs,
                &mut temp_buffer,
                &mut output_buffer,
                &mut properties,
            )?
        };

        if compact {
            stream.synchronize()?;

            let mut compacted_size = 0usize;
            compacted_size_buffer.copy_to(&mut compacted_size)?;

            if compacted_size < sizes.output_size_in_bytes {
                let mut buf = unsafe { DeviceBuffer::<u8>::uninitialized(compacted_size)? };
                let hnd = unsafe { accel_compact(ctx, stream, hnd, &mut buf)? };
                Ok(Accel { buf, hnd })
            } else {
                Ok(Accel {
                    buf: output_buffer,
                    hnd,
                })
            }
        } else {
            Ok(Accel {
                buf: output_buffer,
                hnd,
            })
        }
    }

    /// Construct a new Accel from a handle and buffer.
    pub unsafe fn from_raw_parts(buf: DeviceBuffer<u8>, hnd: TraversableHandle) -> Accel {
        Accel { buf, hnd }
    }

    /// Obtain opaque relocation information for this accel in the given [`DeviceContext`].
    ///
    /// The location information may be passed to
    /// [`check_relocation_compatibility()`](Accel::check_relocation_compatibility) to
    /// determine if this acceleration structure can be relocated to a different device's
    /// memory space.
    ///
    /// When used with [`relocate`](Accel::relocate) it provides the data necessary
    /// for doing the relocation.
    ///
    /// If this acceleration structure is copied multiple times, the same
    /// [`AccelRelocationInfo`] can also be used on all copies.
    pub fn get_relocation_info(&self, ctx: &DeviceContext) -> Result<AccelRelocationInfo> {
        let mut inner = optix_sys::OptixRelocationInfo::default();
        unsafe {
            Ok(optix_call!(optixAccelGetRelocationInfo(
                ctx.raw,
                self.hnd.inner,
                &mut inner
            ))
            .map(|_| AccelRelocationInfo { inner })?)
        }
    }
}

/// Acceleration structure supporting dynamic updates.
///
/// Building an acceleration structure can be computationally costly. Applications
/// may choose to update an existing acceleration structure using modified vertex
/// data or bounding boxes. Updating an existing acceleration structure is generally
/// much faster than rebuilding. However, the quality of the acceleration structure
/// may degrade if the data changes too much with an update, for example, through
/// explosions or other chaotic transitions—even if for only parts of the mesh.
/// The degraded acceleration structure may result in slower traversal performance
/// as compared to an acceleration structure built from scratch from the modified
/// input data.
pub struct DynamicAccel {
    accel: Accel,
    hash: u64,
}

impl Traversable for DynamicAccel {
    /// Get the [`TraversableHandle`] that represents this accel.
    fn handle(&self) -> TraversableHandle {
        self.accel.hnd
    }
}

impl Deref for DynamicAccel {
    type Target = Accel;

    fn deref(&self) -> &Self::Target {
        &self.accel
    }
}

impl DynamicAccel {
    /// Build and compact the acceleration structure for the given inputs.
    ///
    /// This forces the ALLOW_UPDATE flag for the build flags to make sure the
    /// resulting accel can be updated.
    pub fn build<I: BuildInput>(
        ctx: &DeviceContext,
        stream: &cust::stream::Stream,
        accel_options: &mut [AccelBuildOptions],
        build_inputs: &[I],
        compact: bool,
    ) -> Result<DynamicAccel> {
        // Force ALLOW_UPDATE
        for opt in accel_options.iter_mut() {
            opt.build_flags |= BuildFlags::ALLOW_UPDATE;
            opt.operation = BuildOperation::Build;
        }

        let sizes = accel_compute_memory_usage(ctx, accel_options, build_inputs)?;
        let mut output_buffer =
            unsafe { DeviceBuffer::<u8>::uninitialized(sizes.output_size_in_bytes)? };

        let mut temp_buffer =
            unsafe { DeviceBuffer::<u8>::uninitialized(sizes.temp_size_in_bytes)? };

        let compacted_size_buffer = unsafe { DeviceBox::<usize>::uninitialized()? };

        let mut properties = vec![AccelEmitDesc::CompactedSize(
            compacted_size_buffer.as_device_ptr(),
        )];

        let hnd = unsafe {
            accel_build(
                ctx,
                stream,
                accel_options,
                build_inputs,
                &mut temp_buffer,
                &mut output_buffer,
                &mut properties,
            )?
        };

        let mut hasher = DefaultHasher::new();
        build_inputs.hash(&mut hasher);
        let hash = hasher.finish();

        if compact {
            stream.synchronize()?;

            let mut compacted_size = 0usize;
            compacted_size_buffer.copy_to(&mut compacted_size)?;

            let mut buf = unsafe { DeviceBuffer::<u8>::uninitialized(compacted_size)? };

            let hnd = unsafe { accel_compact(ctx, stream, hnd, &mut buf)? };

            Ok(DynamicAccel {
                accel: Accel { buf, hnd },
                hash,
            })
        } else {
            Ok(DynamicAccel {
                accel: Accel {
                    buf: output_buffer,
                    hnd,
                },
                hash,
            })
        }
    }

    /// Update the acceleration structure
    ///
    /// This forces the build operation to Update.
    ///
    /// # Errors
    /// * [`Error::AccelUpdateMismatch`] - if the provided `build_inputs` do
    /// not match the structure of those provided to [`build()`](DynamicAccel::build)
    pub fn update<I: BuildInput>(
        &mut self,
        ctx: &DeviceContext,
        stream: &cust::stream::Stream,
        accel_options: &mut [AccelBuildOptions],
        build_inputs: &[I],
    ) -> Result<()> {
        for opt in accel_options.iter_mut() {
            opt.build_flags |= BuildFlags::ALLOW_UPDATE;
            opt.operation = BuildOperation::Update;
        }

        let mut hasher = DefaultHasher::new();
        build_inputs.hash(&mut hasher);
        let hash = hasher.finish();

        if hash != self.hash {
            return Err(Error::AccelUpdateMismatch);
        }

        let sizes = accel_compute_memory_usage(ctx, accel_options, build_inputs)?;
        let mut output_buffer =
            unsafe { DeviceBuffer::<u8>::uninitialized(sizes.output_size_in_bytes)? };

        let mut temp_buffer =
            unsafe { DeviceBuffer::<u8>::uninitialized(sizes.temp_size_in_bytes)? };

        let compacted_size_buffer = unsafe { DeviceBox::<usize>::uninitialized()? };

        let mut properties = vec![AccelEmitDesc::CompactedSize(
            compacted_size_buffer.as_device_ptr(),
        )];

        let hnd = unsafe {
            accel_build(
                ctx,
                stream,
                accel_options,
                build_inputs,
                &mut temp_buffer,
                &mut output_buffer,
                &mut properties,
            )?
        };

        self.accel = Accel {
            buf: output_buffer,
            hnd,
        };

        Ok(())
    }
}

/// Opaque handle to a traversable acceleration structure.
///
/// # Safety
/// You should consider this handle to be a raw pointer, thus you can copy it
/// and it provides no tracking of lifetime or ownership. You are responsible
/// for ensuring that the device memory containing the acceleration structures
/// this handle references are alive if you try to use this handle
#[repr(transparent)]
#[derive(Copy, Clone, Debug, PartialEq, DeviceCopy, Default)]
pub struct TraversableHandle {
    pub(crate) inner: u64,
}

/// Computes the device memory required for temporary and output buffers
/// when building the acceleration structure. Use the returned sizes to
/// allocate enough memory to pass to [`accel_build()`].
///
/// # Examples
/// ```no_run
/// use cust::prelude as cu;
/// use optix::prelude as ox;
/// # fn doit() -> Result<(), Box<dyn std::error::Error>> {
/// # cust::init(cu::CudaFlags::empty())?;
/// # ox::init()?;
/// # let device = cu::Device::get_device(0)?;
/// # let cu_ctx = cu::Context::create_and_push(cu::ContextFlags::SCHED_AUTO |
/// # cu::ContextFlags::MAP_HOST, device)?;
/// # let ctx = ox::DeviceContext::new(&cu_ctx, false)?;
/// # let vertices: Vec<[f32; 3]> = Vec::new();
/// # let indices: Vec<[u32; 3]> = Vec::new();
/// # let stream = cu::Stream::new(cu::StreamFlags::DEFAULT, None)?;
/// let buf_vertex = DeviceBuffer::from_slice(&vertices)?;
/// let buf_indices = DeviceBuffer::from_slice(&indices)?;
///
/// let geometry_flags = GeometryFlags::None;
/// let build_inputs = [IndexedTriangleArray::new(
///     &[&buf_vertex],
///     &buf_indices,
///     &[geometry_flags],
/// )];
/// let accel_options =
///     AccelBuildOptions::new(BuildFlags::ALLOW_COMPACTION, BuildOperation::Build);
///
/// let sizes = accel_compute_memory_usage(ctx, accel_options, build_inputs)?;
/// let mut output_buffer =
///     unsafe { DeviceBuffer::<u8>::uninitialized(sizes.output_size_in_bytes)? };
///
/// let mut temp_buffer =
///     unsafe { DeviceBuffer::<u8>::uninitialized(sizes.temp_size_in_bytes)? };
///
/// let mut compacted_size_buffer = unsafe { DeviceBox::<usize>::uninitialized()? };
///
/// let mut properties = vec![AccelEmitDesc::CompactedSize(
///     compacted_size_buffer.as_device_ptr(),
/// )];
///
/// let hnd = unsafe {
///     accel_build(
///         ctx,
///         stream,
///         accel_options,
///         build_inputs,
///         &mut temp_buffer,
///         &mut output_buffer,
///         &mut properties,
///     )?
/// };
///
/// # Ok(())
/// # }
/// ```
pub fn accel_compute_memory_usage<I: BuildInput>(
    ctx: &DeviceContext,
    accel_options: &[AccelBuildOptions],
    build_inputs: &[I],
) -> Result<AccelBufferSizes> {
    let mut buffer_sizes = AccelBufferSizes::default();
    let build_sys: Vec<_> = build_inputs.iter().map(|b| b.to_sys()).collect();

    unsafe {
        Ok(optix_call!(optixAccelComputeMemoryUsage(
            ctx.raw,
            accel_options.as_ptr() as *const _,
            build_sys.as_ptr(),
            build_sys.len() as u32,
            &mut buffer_sizes as *mut _ as *mut _,
        ))
        .map(|_| buffer_sizes)?)
    }
}

/// Builds the acceleration structure.
/// `temp_buffer` and `output_buffer` must be at least as large as the sizes
/// returned by `accel_compute_memory_usage()`
///
/// # Examples
/// ```no_run
/// use cust::prelude as cu;
/// use optix::prelude as ox;
/// # fn doit() -> Result<(), Box<dyn std::error::Error>> {
/// # cust::init(cu::CudaFlags::empty())?;
/// # ox::init()?;
/// # let device = cu::Device::get_device(0)?;
/// # let cu_ctx = cu::Context::create_and_push(cu::ContextFlags::SCHED_AUTO |
/// # cu::ContextFlags::MAP_HOST, device)?;
/// # let ctx = ox::DeviceContext::new(&cu_ctx, false)?;
/// # let vertices: Vec<[f32; 3]> = Vec::new();
/// # let indices: Vec<[u32; 3]> = Vec::new();
/// # let stream = cu::Stream::new(cu::StreamFlags::DEFAULT, None)?;
/// let buf_vertex = DeviceBuffer::from_slice(&vertices)?;
/// let buf_indices = DeviceBuffer::from_slice(&indices)?;
///
/// let geometry_flags = GeometryFlags::None;
/// let build_inputs = [IndexedTriangleArray::new(
///     &[&buf_vertex],
///     &buf_indices,
///     &[geometry_flags],
/// )];
/// let accel_options =
///     AccelBuildOptions::new(BuildFlags::ALLOW_COMPACTION, BuildOperation::Build);
///
/// let sizes = accel_compute_memory_usage(ctx, accel_options, build_inputs)?;
/// let mut output_buffer =
///     unsafe { DeviceBuffer::<u8>::uninitialized(sizes.output_size_in_bytes)? };
///
/// let mut temp_buffer =
///     unsafe { DeviceBuffer::<u8>::uninitialized(sizes.temp_size_in_bytes)? };
///
/// let mut compacted_size_buffer = unsafe { DeviceBox::<usize>::uninitialized()? };
///
/// let mut properties = vec![AccelEmitDesc::CompactedSize(
///     compacted_size_buffer.as_device_ptr(),
/// )];
///
/// let hnd = unsafe {
///     accel_build(
///         ctx,
///         stream,
///         accel_options,
///         build_inputs,
///         &mut temp_buffer,
///         &mut output_buffer,
///         &mut properties,
///     )?
/// };
///
/// # Ok(())
/// # }
/// ```
pub unsafe fn accel_build<I: BuildInput>(
    ctx: &DeviceContext,
    stream: &cust::stream::Stream,
    accel_options: &[AccelBuildOptions],
    build_inputs: &[I],
    temp_buffer: &mut DeviceSlice<u8>,
    output_buffer: &mut DeviceSlice<u8>,
    emitted_properties: &mut [AccelEmitDesc],
) -> Result<TraversableHandle> {
    let mut traversable_handle = TraversableHandle { inner: 0 };
    let properties: Vec<optix_sys::OptixAccelEmitDesc> =
        emitted_properties.iter_mut().map(|p| p.into()).collect();

    let build_sys: Vec<_> = build_inputs.iter().map(|b| b.to_sys()).collect();

    Ok(optix_call!(optixAccelBuild(
        ctx.raw,
        stream.as_inner(),
        accel_options.as_ptr() as *const _,
        build_sys.as_ptr(),
        build_sys.len() as u32,
        temp_buffer.as_device_ptr().as_raw(),
        temp_buffer.len(),
        output_buffer.as_device_ptr().as_raw(),
        output_buffer.len(),
        &mut traversable_handle as *mut _ as *mut _,
        properties.as_ptr() as *const _,
        properties.len() as u32,
    ))
    .map(|_| traversable_handle)?)
}

/// Compacts the acceleration structure referenced by `input_handle`,
/// storing the result in `output_buffer` and returning a handle to the
/// newly compacted structure
///
/// # Examples
/// ```no_run
/// use cust::prelude as cu;
/// use optix::prelude as ox;
/// # fn doit() -> Result<(), Box<dyn std::error::Error>> {
/// # cust::init(cu::CudaFlags::empty())?;
/// # ox::init()?;
/// # let device = cu::Device::get_device(0)?;
/// # let cu_ctx = cu::Context::create_and_push(cu::ContextFlags::SCHED_AUTO |
/// # cu::ContextFlags::MAP_HOST, device)?;
/// # let ctx = ox::DeviceContext::new(&cu_ctx, false)?;
/// # let vertices: Vec<[f32; 3]> = Vec::new();
/// # let indices: Vec<[u32; 3]> = Vec::new();
/// # let stream = cu::Stream::new(cu::StreamFlags::DEFAULT, None)?;
/// let buf_vertex = DeviceBuffer::from_slice(&vertices)?;
/// let buf_indices = DeviceBuffer::from_slice(&indices)?;
///
/// let geometry_flags = GeometryFlags::None;
/// let build_inputs = [IndexedTriangleArray::new(
///     &[&buf_vertex],
///     &buf_indices,
///     &[geometry_flags],
/// )];
/// let accel_options =
///     AccelBuildOptions::new(BuildFlags::ALLOW_COMPACTION, BuildOperation::Build);
///
/// let sizes = accel_compute_memory_usage(ctx, accel_options, build_inputs)?;
/// let mut output_buffer =
///     unsafe { DeviceBuffer::<u8>::uninitialized(sizes.output_size_in_bytes)? };
///
/// let mut temp_buffer =
///     unsafe { DeviceBuffer::<u8>::uninitialized(sizes.temp_size_in_bytes)? };
///
/// // Storage for the size of the compacted buffer
/// let mut compacted_size_buffer = unsafe { DeviceBox::<usize>::uninitialized()? };
///
/// // Tell OptiX that we want to know how big the compacted buffer needs to be
/// let mut properties = vec![AccelEmitDesc::CompactedSize(
///     compacted_size_buffer.as_device_ptr(),
/// )];
///
/// let hnd = unsafe {
///     accel_build(
///         ctx,
///         stream,
///         accel_options,
///         build_inputs,
///         &mut temp_buffer,
///         &mut output_buffer,
///         &mut properties,
///     )?
/// };
///
/// // The build is asynchronous, so we need to block on the stream before
/// // reading back the emitted compacted size
/// stream.synchronize()?;
///
/// // Copy the returned size needed for the compacted buffer and allocate
/// // storage
/// let mut compacted_size = 0usize;
/// compacted_size_buffer.copy_to(&mut compacted_size)?;
///
/// let mut buf = unsafe { DeviceBuffer::<u8>::uninitialized(compacted_size)? };
///
/// // Compact the accel structure.
/// let hnd = unsafe { accel_compact(ctx, stream, hnd, &mut buf)? };
///
/// # Ok(())
/// # }
/// ```
pub unsafe fn accel_compact(
    ctx: &DeviceContext,
    stream: &cust::stream::Stream,
    input_handle: TraversableHandle,
    output_buffer: &mut DeviceSlice<u8>,
) -> Result<TraversableHandle> {
    let mut traversable_handle = TraversableHandle { inner: 0 };
    Ok(optix_call!(optixAccelCompact(
        ctx.raw,
        stream.as_inner(),
        input_handle.inner,
        output_buffer.as_device_ptr().as_raw(),
        output_buffer.len(),
        &mut traversable_handle as *mut _ as *mut _,
    ))
    .map(|_| traversable_handle)?)
}

bitflags::bitflags! {
    /// Flags providing configuration options to acceleration structure build.
    ///
    /// * `ALLOW_UPDATE` - Must be provided if the accel is to support dynamic updates.
    /// * `ALLOW_COMPACTION` - Must be provided to enable memory compaction for the accel.
    /// * `PREFER_FAST_TRACE` - Accel build is slower, but tracing against it will be faster.
    /// * `PREFER_FAST_BUILD` - Accel build is faster, but tracing against it will be slower.
    /// * `ALLOW_RANDOM_VERTEX_ACCESS` - Must be provided to be able to get at vertex data from CH
    /// an AH programs on the device. May affect the performance of the accel (seems to be larger).
    ///
    /// Note that `PREFER_FAST_TRACE` and `PREFER_FAST_BUILD` are mutually exclusive.
    #[derive(Default, Clone, Copy, PartialEq, Eq, Debug)]
    pub struct BuildFlags: i32 {
        const NONE = optix_sys::OptixBuildFlags::OPTIX_BUILD_FLAG_NONE as i32;
        const ALLOW_UPDATE = optix_sys::OptixBuildFlags::OPTIX_BUILD_FLAG_ALLOW_UPDATE as i32;
        const ALLOW_COMPACTION = optix_sys::OptixBuildFlags::OPTIX_BUILD_FLAG_ALLOW_COMPACTION as i32;
        const PREFER_FAST_TRACE = optix_sys::OptixBuildFlags::OPTIX_BUILD_FLAG_PREFER_FAST_TRACE as i32;
        const PREFER_FAST_BUILD = optix_sys::OptixBuildFlags::OPTIX_BUILD_FLAG_PREFER_FAST_BUILD as i32;
        const ALLOW_RANDOM_VERTEX_ACCESS = optix_sys::OptixBuildFlags::OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS as i32;
    }
}

/// Select which operation to perform with [`accel_build()`].
#[repr(i32)]
#[derive(Debug, Copy, Clone, PartialEq, Default)]
pub enum BuildOperation {
    #[default]
    Build = optix_sys::OptixBuildOperation::OPTIX_BUILD_OPERATION_BUILD as i32,
    Update = optix_sys::OptixBuildOperation::OPTIX_BUILD_OPERATION_UPDATE as i32,
}

/// Configure how to handle ray times that are outside of the provided motion keys.
///
/// By default, the object will appear static (clamped) to the nearest motion
/// key for rays outside of the range of key times.
///
/// * `START_VANISH` - The object will be invisible to rays with a time less
/// than the first provided motion key
/// * `END_VANISH` - The object will be invisible to rays with a time less
/// than the first provided motion key
#[derive(DeviceCopy, Clone, Copy, PartialEq, Eq, Debug)]
pub struct MotionFlags(u16);

bitflags::bitflags! {
   impl MotionFlags: u16 {
        const NONE = optix_sys::OptixMotionFlags::OPTIX_MOTION_FLAG_NONE as u16;
        const START_VANISH = optix_sys::OptixMotionFlags::OPTIX_MOTION_FLAG_START_VANISH as u16;
        const END_VANISH = optix_sys::OptixMotionFlags::OPTIX_MOTION_FLAG_END_VANISH as u16;
    }
}

/// Provide an accel build with motion keys for motion blur.
///
/// The motion options are always specified per traversable (acceleration structure
/// or motion transform). There is no dependency between the motion options of
/// traversables; given an instance referencing a geometry acceleration structure
/// with motion, it is not required to build an instance acceleration structure
/// with motion. The same goes for motion transforms. Even if an instance references
/// a motion transform as child traversable, the instance acceleration structure
/// itself may or may not have motion.
///
/// Motion transforms must specify at least two motion keys. Acceleration structures,
/// however, also accept [`BuildOptions`] with field `motion_options` set
/// to zero. This effectively disables motion for the acceleration structure and
/// ignores the motion beginning and ending times, along with the motion flags.
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, DeviceCopy)]
pub struct MotionOptions {
    pub num_keys: u16,
    pub flags: MotionFlags,
    pub time_begin: f32,
    pub time_end: f32,
}

impl Default for MotionOptions {
    fn default() -> Self {
        MotionOptions {
            num_keys: 0,
            flags: MotionFlags::NONE,
            time_begin: 0.0,
            time_end: 0.0,
        }
    }
}

const_assert_eq!(
    std::mem::size_of::<MotionOptions>(),
    std::mem::size_of::<optix_sys::OptixMotionOptions>(),
);

/// Options to configure the [`accel_build()`]
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Default)]
pub struct AccelBuildOptions {
    build_flags: BuildFlags,
    operation: BuildOperation,
    motion_options: MotionOptions,
}

impl AccelBuildOptions {
    /// Create a new AccelBuildOptions with the given flags and operation and
    /// no motion blur.
    pub fn new(build_flags: BuildFlags) -> Self {
        AccelBuildOptions {
            build_flags,
            operation: BuildOperation::Build,
            motion_options: MotionOptions {
                num_keys: 1,
                flags: MotionFlags::NONE,
                time_begin: 0.0f32,
                time_end: 1.0f32,
            },
        }
    }

    /// Set the build operation to either build or update
    pub fn build_operation(mut self, op: BuildOperation) -> Self {
        self.operation = op;
        self
    }

    /// Set the number of motion keys.
    ///
    /// This must either be 0 for no motion blur, or >= 2.
    pub fn num_keys(mut self, num_keys: u16) -> Self {
        self.motion_options.num_keys = num_keys;
        self
    }

    /// Set the start and end time that the first and last motion keys represent.
    pub fn time_interval(mut self, time_begin: f32, time_end: f32) -> Self {
        self.motion_options.time_begin = time_begin;
        self.motion_options.time_end = time_end;
        self
    }

    /// Set the flags describing how to handle out-of-range time samples.
    pub fn motion_flags(mut self, flags: MotionFlags) -> Self {
        self.motion_options.flags = flags;
        self
    }
}

/// Opaque relocation information for an [`Accel`] in a given [`DeviceContext`].
///
/// The location information may be passed to
/// [`check_relocation_compatibility()`](Accel::check_relocation_compatibility) to
/// determine if the associated acceleration structure can be relocated to a different device's
/// memory space.
///
/// When used with [`relocate`](Accel::relocate) it provides the data necessary
/// for doing the relocation.
///
/// If the acceleration structure is copied multiple times, the same
/// [`AccelRelocationInfo`] can also be used on all copies.
#[repr(transparent)]
pub struct AccelRelocationInfo {
    #[allow(dead_code)]
    inner: optix_sys::OptixRelocationInfo,
}

/// Struct used for OptiX to communicate the necessary buffer sizes for accel
/// temp and final outputs.
#[repr(C)]
#[derive(Debug, Default, Copy, Clone, PartialEq)]
pub struct AccelBufferSizes {
    pub output_size_in_bytes: usize,
    pub temp_size_in_bytes: usize,
    pub temp_update_size_in_bytes: usize,
}

/// Struct used for Optix to communicate the compacted size or list of bounding
/// boxes back from an accel build.
///
/// # Examples
/// ```
/// // Copy the returned size needed for the compacted buffer and allocate
/// // storage
/// let mut compacted_size = 0usize;
/// compacted_size_buffer.copy_to(&mut compacted_size)?;
///
/// let mut buf = unsafe { DeviceBuffer::<u8>::uninitialized(compacted_size)? };
///
/// // Compact the accel structure.
/// let hnd = unsafe { accel_compact(ctx, stream, hnd, &mut buf)? };
/// ```
pub enum AccelEmitDesc {
    CompactedSize(DevicePointer<usize>),
    Aabbs(DevicePointer<Aabb>),
}

/// An axis-aligned bounding box.
///
/// Used to communicate bounds info to and from OptiX for bounding custom primitives
/// and instances
#[repr(C)]
#[derive(Debug, DeviceCopy, Copy, Clone)]
pub struct Aabb {
    min: Vector3<f32>,
    max: Vector3<f32>,
}

impl Aabb {
    /// Create a new Aabb by supplying the min and max points
    pub fn new<V: Into<Vector3<f32>>>(min: V, max: V) -> Self {
        let min = min.into();
        let max = max.into();
        Self { min, max }
    }
}

impl From<&mut AccelEmitDesc> for optix_sys::OptixAccelEmitDesc {
    fn from(aed: &mut AccelEmitDesc) -> Self {
        match aed {
            AccelEmitDesc::CompactedSize(p) => Self {
                result: p.as_raw(),
                type_: optix_sys::OptixAccelPropertyType::OPTIX_PROPERTY_TYPE_COMPACTED_SIZE,
            },
            AccelEmitDesc::Aabbs(p) => Self {
                result: p.as_raw(),
                type_: optix_sys::OptixAccelPropertyType::OPTIX_PROPERTY_TYPE_AABBS,
            },
        }
    }
}

/// Per-geometry tracing requirements used to allow potential optimizations.
///
/// * `GeometryFlags::None` - Applies the default behavior when calling the
///     any-hit program, possibly multiple times, allowing the acceleration-structure
///     builder to apply all optimizations.
/// * `GeometryFlags::DisableAnyHit` - Disables some optimizations specific to
///     acceleration-structure builders. By default, traversal may call the any-hit
///     program more than once for each intersected primitive. Setting the flag
///     ensures that the any-hit program is called only once for a hit with a primitive.
///     However, setting this flag may change traversal performance. The usage of
///     this flag may be required for correctness of some rendering algorithms;
///     for example, in cases where opacity or transparency information is accumulated
///     in an any-hit program.
/// * `GeometryFlags::RequireSingleAnyHitCall` - Indicates that traversal should
///     not call the any-hit program for this primitive even if the corresponding SBT
///     record contains an any-hit program. Setting this flag usually improves
///     performance even if no any-hit program is present in the SBT.
#[repr(u32)]
#[derive(Copy, Clone, PartialEq, Hash)]
pub enum GeometryFlags {
    None = optix_sys::OptixGeometryFlags::None as u32,
    DisableAnyHit = optix_sys::OptixGeometryFlags::DisableAnyHit as u32,
    RequireSingleAnyHitCall = optix_sys::OptixGeometryFlags::RequireSingleAnyHitCall as u32,
}

impl From<GeometryFlags> for optix_sys::OptixGeometryFlags {
    fn from(f: GeometryFlags) -> Self {
        match f {
            GeometryFlags::None => optix_sys::OptixGeometryFlags::None,
            GeometryFlags::DisableAnyHit => optix_sys::OptixGeometryFlags::DisableAnyHit,
            GeometryFlags::RequireSingleAnyHitCall => {
                optix_sys::OptixGeometryFlags::RequireSingleAnyHitCall
            }
        }
    }
}

impl From<GeometryFlags> for u32 {
    fn from(f: GeometryFlags) -> Self {
        match f {
            GeometryFlags::None => optix_sys::OptixGeometryFlags::None as u32,
            GeometryFlags::DisableAnyHit => optix_sys::OptixGeometryFlags::DisableAnyHit as u32,
            GeometryFlags::RequireSingleAnyHitCall => {
                optix_sys::OptixGeometryFlags::RequireSingleAnyHitCall as u32
            }
        }
    }
}

/// Specify acceleration structure build input data for a curves geometry
///
/// A curve is a swept surface defined by a 3D spline curve and a varying width (radius). A curve (or "strand") of degree d (3=cubic, 2=quadratic, 1=linear) is represented by N > d vertices and N width values, and comprises N - d segments. Each segment is defined by d+1 consecutive vertices. Each curve may have a different number of vertices.
///
/// OptiX describes the curve array as a list of curve segments. The primitive id is the segment number. It is the user's responsibility to maintain a mapping between curves and curve segments. Each index buffer entry i = indexBuffer[primid] specifies the start of a curve segment, represented by d+1 consecutive vertices in the vertex buffer, and d+1 consecutive widths in the width buffer. Width is interpolated the same way vertices are interpolated, that is, using the curve basis.
///
/// Each curves build input has only one SBT record. To create curves with different materials in the same BVH, use multiple build inputs.
pub struct CurveArray<'v, 'w, 'i> {
    curve_type: CurveType,
    num_primitives: u32,
    vertex_buffers: PhantomData<&'v f32>,
    num_vertices: u32,
    d_vertex_buffers: Vec<CUdeviceptr>,
    vertex_stride_in_bytes: u32,
    width_buffers: PhantomData<&'w f32>,
    #[allow(dead_code)]
    num_width_buffers: u32,
    d_width_buffers: Vec<CUdeviceptr>,
    width_stride_in_bytes: u32,
    index_buffer: &'i DeviceSlice<u32>,
    index_stride_in_bytes: u32,
    flags: GeometryFlags,
    primitive_index_offset: u32,
}

impl Hash for CurveArray<'_, '_, '_> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.curve_type.hash(state);
        state.write_u32(self.num_primitives);
        state.write_u32(self.num_vertices);
        state.write_usize(self.d_vertex_buffers.len());
        state.write_u32(self.vertex_stride_in_bytes);
        state.write_u32(self.num_vertices);
        state.write_usize(self.d_width_buffers.len());
        state.write_u32(self.width_stride_in_bytes);
        state.write_usize(self.index_buffer.len());
        state.write_u32(self.index_stride_in_bytes);
        self.flags.hash(state);
        state.write_u32(self.primitive_index_offset);
    }
}

impl<'v, 'w, 'i> CurveArray<'v, 'w, 'i> {
    /// Constructor
    ///
    /// # Parameters
    /// * `curve_type` - Curve degree and basis
    /// * `vertex_buffers` - A slice of device buffers, one per motion step.
    ///     The length of this slice must match the number of motion keys specified
    ///     in [`AccelBuildOptions::motion_options`]
    /// * `width_buffers` - Parallel to `vertex_buffers` with matching lengths and
    ///     number of motion steps. One value per vertex specifying the width of
    ///     the curve
    /// * `index_buffer` - An array of u32, one per curve segment. Each index is
    ///     the start of `degree+1` consecutive vertices in `vertex_buffers`, and
    ///     corresponding widths in `width_buffers`. These define a single segment.
    ///     The length of this array is therefore the number of curve segments
    pub fn new(
        curve_type: CurveType,
        vertex_buffers: &[&'v DeviceSlice<f32>],
        width_buffers: &[&'w DeviceSlice<f32>],
        index_buffer: &'i DeviceSlice<u32>,
    ) -> Result<CurveArray<'v, 'w, 'i>> {
        // TODO (AL): Do some sanity checking on the values here
        let num_vertices = vertex_buffers[0].len() as u32;
        let d_vertex_buffers: Vec<_> = vertex_buffers
            .iter()
            .map(|b| b.as_device_ptr().as_raw())
            .collect();

        let num_width_buffers = width_buffers.len() as u32;
        let d_width_buffers: Vec<_> = width_buffers
            .iter()
            .map(|b| b.as_device_ptr().as_raw())
            .collect();

        Ok(CurveArray {
            curve_type,
            num_primitives: index_buffer.len() as u32,
            vertex_buffers: PhantomData,
            num_vertices,
            d_vertex_buffers,
            vertex_stride_in_bytes: 0,
            width_buffers: PhantomData,
            num_width_buffers,
            d_width_buffers,
            width_stride_in_bytes: 0,
            index_buffer,
            index_stride_in_bytes: 0,
            flags: GeometryFlags::None,
            primitive_index_offset: 0,
        })
    }

    /// Stride between vertices. If not specified, vertices are assumed to be
    /// tightly packed.
    pub fn vertex_stride(mut self, stride_in_bytes: u32) -> Self {
        self.vertex_stride_in_bytes = stride_in_bytes;
        self
    }

    /// Stride between width values. If not specified, values are assumed to be
    /// tightly packed.
    pub fn width_stride(mut self, stride_in_bytes: u32) -> Self {
        self.vertex_stride_in_bytes = stride_in_bytes;
        self
    }

    /// Stride between indices. If not specified, indices are assumed to be
    /// tightly packed.
    pub fn index_stride(mut self, stride_in_bytes: u32) -> Self {
        self.vertex_stride_in_bytes = stride_in_bytes;
        self
    }

    /// Combination of [`GeometryFlags`] specifying the primitive behaviour
    pub fn flags(mut self, flags: GeometryFlags) -> Self {
        self.flags = flags;
        self
    }

    /// Primitive index bias, applied on the device in `optixGetPrimitiveIndex()`.
    ///
    /// Sum of primitiveIndexOffset and number of primitives must not overflow 32bits.
    pub fn primitive_index_offset(mut self, offset: u32) -> Self {
        self.primitive_index_offset = offset;
        self
    }
}

impl BuildInput for CurveArray<'_, '_, '_> {
    fn to_sys(&self) -> optix_sys::OptixBuildInput {
        let mut v = optix_sys::OptixBuildInput {
            type_: optix_sys::OptixBuildInputType::OPTIX_BUILD_INPUT_TYPE_CURVES,
            ..Default::default()
        };
        unsafe {
            *v.__bindgen_anon_1.curveArray.as_mut() = optix_sys::OptixBuildInputCurveArray {
                curveType: self.curve_type.into(),
                numPrimitives: self.num_primitives,
                vertexBuffers: self.d_vertex_buffers.as_ptr() as *const CUdeviceptr,
                numVertices: self.num_vertices,
                vertexStrideInBytes: self.vertex_stride_in_bytes,
                widthBuffers: self.d_width_buffers.as_ptr() as *const CUdeviceptr,
                widthStrideInBytes: self.width_stride_in_bytes,
                normalBuffers: std::ptr::null(),
                normalStrideInBytes: 0,
                indexBuffer: self.index_buffer.as_device_ptr().as_raw(),
                indexStrideInBytes: self.index_stride_in_bytes,
                flag: self.flags as u32,
                primitiveIndexOffset: self.primitive_index_offset,
                endcapFlags: optix_sys::OptixCurveEndcapFlags::OPTIX_CURVE_ENDCAP_DEFAULT as u32,
            };
        };
        v
    }
}

/// Specifies the type of curves, either linear, quadratic or cubic b-splines.
#[derive(Debug, Copy, Clone, PartialEq, Hash)]
pub enum CurveType {
    RoundLinear,
    RoundQuadraticBSpline,
    RoundCubicBSpline,
}

impl From<CurveType> for optix_sys::OptixPrimitiveType {
    fn from(c: CurveType) -> Self {
        match c {
            CurveType::RoundLinear => {
                optix_sys::OptixPrimitiveType::OPTIX_PRIMITIVE_TYPE_ROUND_LINEAR
            }
            CurveType::RoundQuadraticBSpline => {
                optix_sys::OptixPrimitiveType::OPTIX_PRIMITIVE_TYPE_ROUND_QUADRATIC_BSPLINE
            }
            CurveType::RoundCubicBSpline => {
                optix_sys::OptixPrimitiveType::OPTIX_PRIMITIVE_TYPE_ROUND_CUBIC_BSPLINE
            }
        }
    }
}

/// Specifies the type of vertex data
#[repr(i32)]
#[derive(Copy, Clone, PartialEq)]
pub enum VertexFormat {
    None = optix_sys::OptixVertexFormat_OPTIX_VERTEX_FORMAT_NONE as i32,
    Float3 = optix_sys::OptixVertexFormat_OPTIX_VERTEX_FORMAT_FLOAT3 as i32,
    Float2 = optix_sys::OptixVertexFormat_OPTIX_VERTEX_FORMAT_FLOAT2 as i32,
    Half3 = optix_sys::OptixVertexFormat_OPTIX_VERTEX_FORMAT_HALF3 as i32,
    Half2 = optix_sys::OptixVertexFormat_OPTIX_VERTEX_FORMAT_HALF2 as i32,
    SNorm16 = optix_sys::OptixVertexFormat_OPTIX_VERTEX_FORMAT_SNORM16_3 as i32,
    SNorm32 = optix_sys::OptixVertexFormat_OPTIX_VERTEX_FORMAT_SNORM16_2 as i32,
}

/// Specifies the type of index data
#[repr(i32)]
#[derive(Copy, Clone, PartialEq)]
pub enum IndicesFormat {
    None = optix_sys::OptixIndicesFormat_OPTIX_INDICES_FORMAT_NONE as i32,
    Short3 = optix_sys::OptixIndicesFormat_OPTIX_INDICES_FORMAT_UNSIGNED_SHORT3 as i32,
    Int3 = optix_sys::OptixIndicesFormat_OPTIX_INDICES_FORMAT_UNSIGNED_INT3 as i32,
}

/// Specifies the format of transform data
#[repr(i32)]
#[derive(Copy, Clone, PartialEq)]
pub enum TransformFormat {
    None = optix_sys::OptixTransformFormat::OPTIX_TRANSFORM_FORMAT_NONE as i32,
    MatrixFloat12 = optix_sys::OptixTransformFormat::OPTIX_TRANSFORM_FORMAT_MATRIX_FLOAT12 as i32,
}

/// Trait allowing the triangle builds to be generic over the input vertex data.
///
/// For instance, if you had a custom vertex type:
/// ```
/// struct MyVertex {
///      x: i16,
///      y: i16,
///      z: i16,
///      nx: f32,
///      ny: f32,
///      nz: f32,
/// }
///
/// impl Vertex for MyVertex {
///     const FORMAT: VertexFormat = VertexFormat::SNorm16;
///     const STRIDE: u32 = 18;
/// }
/// ```
pub trait Vertex: cust::memory::DeviceCopy {
    const FORMAT: VertexFormat;
    const STRIDE: u32 = 0;
}

#[cfg(feature = "impl_half")]
impl Vertex for [half::f16; 2] {
    const FORMAT: VertexFormat = VertexFormat::Half2;
}

#[cfg(feature = "impl_half")]
impl Vertex for [half::f16; 3] {
    const FORMAT: VertexFormat = VertexFormat::Half3;
}

#[cfg(feature = "impl_half")]
impl Vertex for mint::Vector2<half::f16> {
    const FORMAT: VertexFormat = VertexFormat::Half2;
}

#[cfg(feature = "impl_half")]
impl Vertex for mint::Vector3<half::f16> {
    const FORMAT: VertexFormat = VertexFormat::Half3;
}

impl Vertex for [f32; 2] {
    const FORMAT: VertexFormat = VertexFormat::Float2;
}

impl Vertex for [f32; 3] {
    const FORMAT: VertexFormat = VertexFormat::Float3;
}

impl Vertex for [i16; 3] {
    const FORMAT: VertexFormat = VertexFormat::SNorm16;
}

impl Vertex for [i32; 3] {
    const FORMAT: VertexFormat = VertexFormat::SNorm32;
}

impl Vertex for mint::Vector2<f32> {
    const FORMAT: VertexFormat = VertexFormat::Float2;
}

impl Vertex for mint::Vector3<f32> {
    const FORMAT: VertexFormat = VertexFormat::Float3;
}

impl Vertex for mint::Vector3<i16> {
    const FORMAT: VertexFormat = VertexFormat::SNorm16;
}

impl Vertex for mint::Vector3<i32> {
    const FORMAT: VertexFormat = VertexFormat::SNorm32;
}

/// Trait allowing build inputs to be generic over the index type
pub trait IndexTriple: cust::memory::DeviceCopy {
    const FORMAT: IndicesFormat;
    const STRIDE: u32 = 0;
}

impl IndexTriple for [u16; 3] {
    const FORMAT: IndicesFormat = IndicesFormat::Short3;
}

impl IndexTriple for [u32; 3] {
    const FORMAT: IndicesFormat = IndicesFormat::Int3;
}

impl IndexTriple for mint::Vector3<u16> {
    const FORMAT: IndicesFormat = IndicesFormat::Short3;
}

impl IndexTriple for mint::Vector3<u32> {
    const FORMAT: IndicesFormat = IndicesFormat::Int3;
}

/// Build input for specifying a (non-indexed) triangle geometry
pub struct TriangleArray<'v, 'g, V: Vertex> {
    // We hold slices here to make sure the referenced device memory remains
    // valid for the lifetime of the build input
    vertex_buffers: PhantomData<&'v V>,
    num_vertices: u32,
    d_vertex_buffers: Vec<CUdeviceptr>,
    // per-sbt-record geometry flags
    geometry_flags: &'g [GeometryFlags],
    pre_transform: Option<DevicePointer<[f32; 12]>>,
}

impl<'v, 'g, V: Vertex> TriangleArray<'v, 'g, V> {
    pub fn new(vertex_buffers: &[&'v DeviceSlice<V>], geometry_flags: &'g [GeometryFlags]) -> Self {
        // TODO (AL): do some sanity checking on the slice lengths here
        let num_vertices = vertex_buffers[0].len() as u32;
        let d_vertex_buffers: Vec<_> = vertex_buffers
            .iter()
            .map(|b| b.as_device_ptr().as_raw())
            .collect();
        TriangleArray {
            vertex_buffers: PhantomData,
            num_vertices,
            d_vertex_buffers,
            geometry_flags,
            pre_transform: None,
        }
    }

    pub fn pre_transform(mut self, pre_transform: DevicePointer<[f32; 12]>) -> Self {
        self.pre_transform = Some(pre_transform);
        self
    }
}

impl<V: Vertex> Hash for TriangleArray<'_, '_, V> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        state.write_u32(self.num_vertices);
        state.write_usize(self.d_vertex_buffers.len());
        self.geometry_flags.hash(state);
    }
}

impl<V: Vertex> BuildInput for TriangleArray<'_, '_, V> {
    fn to_sys(&self) -> optix_sys::OptixBuildInput {
        let mut v = optix_sys::OptixBuildInput {
            type_: optix_sys::OptixBuildInputType::OPTIX_BUILD_INPUT_TYPE_TRIANGLES,
            ..Default::default()
        };
        unsafe {
            *v.__bindgen_anon_1.triangleArray.as_mut() = optix_sys::OptixBuildInputTriangleArray {
                vertexBuffers: self.d_vertex_buffers.as_ptr(),
                numVertices: self.num_vertices,
                vertexFormat: V::FORMAT as _,
                vertexStrideInBytes: V::STRIDE,
                indexBuffer: 0,
                numIndexTriplets: 0,
                indexFormat: 0,
                indexStrideInBytes: 0,
                flags: self.geometry_flags.as_ptr() as *const _,
                numSbtRecords: 1,
                sbtIndexOffsetBuffer: 0,
                sbtIndexOffsetSizeInBytes: 0,
                sbtIndexOffsetStrideInBytes: 0,
                primitiveIndexOffset: 0,
                preTransform: if let Some(t) = self.pre_transform {
                    t.as_raw()
                } else {
                    0
                },
                transformFormat: if self.pre_transform.is_some() {
                    optix_sys::OptixTransformFormat::OPTIX_TRANSFORM_FORMAT_MATRIX_FLOAT12
                } else {
                    optix_sys::OptixTransformFormat::OPTIX_TRANSFORM_FORMAT_NONE
                },
                opacityMicromap: optix_sys::OptixBuildInputOpacityMicromap::default(),
                displacementMicromap: optix_sys::OptixBuildInputDisplacementMicromap::default(),
            };
        };
        v
    }
}

pub struct IndexedTriangleArray<'v, 'i, V: Vertex, I: IndexTriple> {
    // We hold slices here to make sure the referenced device memory remains
    // valid for the lifetime of the build input
    vertex_buffers: PhantomData<&'v V>,
    num_vertices: u32,
    d_vertex_buffers: Vec<CUdeviceptr>,
    index_buffer: &'i DeviceSlice<I>,
    // per-object geometry flags
    geometry_flags: Vec<GeometryFlags>,
    pre_transform: Option<DevicePointer<[f32; 12]>>,
}

impl<'v, 'i, V: Vertex, I: IndexTriple> IndexedTriangleArray<'v, 'i, V, I> {
    pub fn new(
        vertex_buffers: &[&'v DeviceSlice<V>],
        index_buffer: &'i DeviceSlice<I>,
        geometry_flags: &[GeometryFlags],
    ) -> Self {
        let num_vertices = vertex_buffers[0].len() as u32;
        let d_vertex_buffers: Vec<_> = vertex_buffers
            .iter()
            .map(|b| b.as_device_ptr().as_raw())
            .collect();
        IndexedTriangleArray {
            vertex_buffers: PhantomData,
            num_vertices,
            d_vertex_buffers,
            geometry_flags: geometry_flags.to_vec(),
            index_buffer,
            pre_transform: None,
        }
    }

    pub fn pre_transform(mut self, pre_transform: DevicePointer<[f32; 12]>) -> Self {
        self.pre_transform = Some(pre_transform);
        self
    }
}

impl<V: Vertex, I: IndexTriple> Hash for IndexedTriangleArray<'_, '_, V, I> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        state.write_u32(self.num_vertices);
        state.write_usize(self.d_vertex_buffers.len());
        self.geometry_flags.hash(state);
        state.write_usize(self.index_buffer.len());
    }
}

impl<V: Vertex, I: IndexTriple> BuildInput for IndexedTriangleArray<'_, '_, V, I> {
    fn to_sys(&self) -> optix_sys::OptixBuildInput {
        let mut v = optix_sys::OptixBuildInput {
            type_: optix_sys::OptixBuildInputType::OPTIX_BUILD_INPUT_TYPE_TRIANGLES,
            ..Default::default()
        };
        unsafe {
            *v.__bindgen_anon_1.triangleArray.as_mut() = optix_sys::OptixBuildInputTriangleArray {
                vertexBuffers: self.d_vertex_buffers.as_ptr(),
                numVertices: self.num_vertices,
                vertexFormat: V::FORMAT as _,
                vertexStrideInBytes: V::STRIDE,
                indexBuffer: self.index_buffer.as_device_ptr().as_raw(),
                numIndexTriplets: self.index_buffer.len() as u32,
                indexFormat: I::FORMAT as _,
                indexStrideInBytes: I::STRIDE,
                flags: self.geometry_flags.as_ptr() as *const _,
                numSbtRecords: 1,
                sbtIndexOffsetBuffer: 0,
                sbtIndexOffsetSizeInBytes: 0,
                sbtIndexOffsetStrideInBytes: 0,
                primitiveIndexOffset: 0,
                preTransform: if let Some(t) = self.pre_transform {
                    t.as_raw()
                } else {
                    0
                },
                transformFormat: if self.pre_transform.is_some() {
                    optix_sys::OptixTransformFormat::OPTIX_TRANSFORM_FORMAT_MATRIX_FLOAT12
                } else {
                    optix_sys::OptixTransformFormat::OPTIX_TRANSFORM_FORMAT_NONE
                },
                opacityMicromap: optix_sys::OptixBuildInputOpacityMicromap::default(),
                displacementMicromap: optix_sys::OptixBuildInputDisplacementMicromap::default(),
            };
        };
        v
    }
}

pub struct CustomPrimitiveArray<'a, 's> {
    aabb_buffers: Vec<CUdeviceptr>,
    aabb_buffers_marker: PhantomData<&'a Aabb>,
    num_primitives: u32,
    stride_in_bytes: u32,
    flags: Vec<GeometryFlags>,
    num_sbt_records: u32,
    sbt_index_offset_buffer: Option<&'s DeviceSlice<u32>>,
    sbt_index_offset_stride_in_bytes: u32,
    primitive_index_offset: u32,
}

impl Hash for CustomPrimitiveArray<'_, '_> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        state.write_usize(self.aabb_buffers.len());
        state.write_u32(self.num_primitives);
        state.write_u32(self.stride_in_bytes);
        self.flags.hash(state);
        state.write_u32(self.num_sbt_records);
        if let Some(b) = self.sbt_index_offset_buffer {
            state.write_usize(b.len());
        } else {
            state.write_usize(0);
        }
        state.write_u32(self.sbt_index_offset_stride_in_bytes);
        state.write_u32(self.primitive_index_offset);
    }
}

impl<'a, 's> CustomPrimitiveArray<'a, 's> {
    pub fn new(
        aabb_buffers: &[&'a DeviceSlice<Aabb>],
        flags: &[GeometryFlags],
    ) -> Result<CustomPrimitiveArray<'a, 's>> {
        let num_primitives = aabb_buffers.len() as u32;
        let aabb_buffers: Vec<_> = aabb_buffers
            .iter()
            .map(|b| b.as_device_ptr().as_raw())
            .collect();

        Ok(CustomPrimitiveArray {
            aabb_buffers,
            aabb_buffers_marker: PhantomData,
            num_primitives,
            stride_in_bytes: 0,
            flags: flags.to_vec(),
            num_sbt_records: 1,
            sbt_index_offset_buffer: None,
            sbt_index_offset_stride_in_bytes: 0,
            primitive_index_offset: 0,
        })
    }

    pub fn stride(mut self, stride_in_bytes: u32) -> Self {
        self.stride_in_bytes = stride_in_bytes;
        self
    }

    pub fn primitive_index_offset(mut self, offset: u32) -> Self {
        self.primitive_index_offset = offset;
        self
    }

    pub fn num_sbt_records(mut self, num_sbt_records: u32) -> Self {
        self.num_sbt_records = num_sbt_records;
        self
    }

    pub fn sbt_index_offset_buffer(
        mut self,
        sbt_index_offset_buffer: &'s DeviceSlice<u32>,
    ) -> Self {
        self.sbt_index_offset_buffer = Some(sbt_index_offset_buffer);
        self
    }

    pub fn sbt_index_offset_buffer_stride(mut self, stride_in_bytes: u32) -> Self {
        self.sbt_index_offset_stride_in_bytes = stride_in_bytes;
        self
    }
}

impl BuildInput for CustomPrimitiveArray<'_, '_> {
    fn to_sys(&self) -> optix_sys::OptixBuildInput {
        let mut v = optix_sys::OptixBuildInput {
            type_: optix_sys::OptixBuildInputType::OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES,
            ..Default::default()
        };
        unsafe {
            *v.__bindgen_anon_1.customPrimitiveArray.as_mut() =
                optix_sys::OptixBuildInputCustomPrimitiveArray {
                    aabbBuffers: self.aabb_buffers.as_ptr(),
                    numPrimitives: self.num_primitives,
                    strideInBytes: self.stride_in_bytes,
                    flags: self.flags.as_ptr() as *const u32,
                    numSbtRecords: self.num_sbt_records,
                    sbtIndexOffsetBuffer: if let Some(sbt_index_offset_buffer) =
                        self.sbt_index_offset_buffer
                    {
                        sbt_index_offset_buffer.as_device_ptr().as_raw()
                    } else {
                        0
                    },
                    sbtIndexOffsetSizeInBytes: 4,
                    sbtIndexOffsetStrideInBytes: self.sbt_index_offset_stride_in_bytes,
                    primitiveIndexOffset: self.primitive_index_offset,
                };
        };
        v
    }
}

#[repr(C, align(16))]
#[derive(Debug, Copy, Clone, DeviceCopy)]
pub struct Instance<'a> {
    transform: RowMatrix3x4<f32>,
    instance_id: u32,
    sbt_offset: u32,
    visibility_mask: u32,
    flags: InstanceFlags,
    traversable_handle: TraversableHandle,
    pad: [u32; 2],
    accel: PhantomData<&'a ()>,
}

const_assert_eq!(
    std::mem::align_of::<Instance>(),
    optix_sys::OptixInstanceByteAlignment
);
const_assert_eq!(
    std::mem::size_of::<Instance>(),
    std::mem::size_of::<optix_sys::OptixInstance>()
);

#[derive(DeviceCopy, Clone, Copy, PartialEq, Eq, Debug)]
pub struct InstanceFlags(i32);
bitflags::bitflags! {
    impl InstanceFlags: i32 {
        const NONE = optix_sys::OptixInstanceFlags::OPTIX_INSTANCE_FLAG_NONE as i32;
        const DISABLE_TRIANGLE_FACE_CULLING = optix_sys::OptixInstanceFlags::OPTIX_INSTANCE_FLAG_DISABLE_TRIANGLE_FACE_CULLING as i32;
        const FLIP_TRIANGLE_FACING = optix_sys::OptixInstanceFlags::OPTIX_INSTANCE_FLAG_FLIP_TRIANGLE_FACING as i32;
        const DISABLE_ANYHIT = optix_sys::OptixInstanceFlags::OPTIX_INSTANCE_FLAG_DISABLE_ANYHIT as i32;
        const ENFORCE_ANYHIT = optix_sys::OptixInstanceFlags::OPTIX_INSTANCE_FLAG_ENFORCE_ANYHIT as i32;
        const FORCE_OPACITY_MICROMAP_2_STATE = optix_sys::OptixInstanceFlags::OPTIX_INSTANCE_FLAG_FORCE_OPACITY_MICROMAP_2_STATE as i32;
        const DISABLE_OPACITY_MICROMAPS = optix_sys::OptixInstanceFlags::OPTIX_INSTANCE_FLAG_DISABLE_OPACITY_MICROMAPS as i32;
    }
}

impl<'a> Instance<'a> {
    pub fn new<T: Traversable>(accel: &'a T) -> Instance<'a> {
        #[allow(clippy::deprecated_cfg_attr)]
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
            traversable_handle: accel.handle(),
            pad: [0; 2],
            accel: PhantomData,
        }
    }

    pub unsafe fn from_handle(traversable_handle: TraversableHandle) -> Instance<'static> {
        #[allow(clippy::deprecated_cfg_attr)]
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
            accel: PhantomData,
        }
    }

    pub fn transform<T: Into<RowMatrix3x4<f32>>>(mut self, transform: T) -> Instance<'a> {
        self.transform = transform.into();
        self
    }

    pub fn instance_id(mut self, instance_id: u32) -> Instance<'a> {
        self.instance_id = instance_id;
        self
    }

    pub fn sbt_offset(mut self, sbt_offset: u32) -> Instance<'a> {
        self.sbt_offset = sbt_offset;
        self
    }

    pub fn visibility_mask(mut self, visibility_mask: u8) -> Instance<'a> {
        self.visibility_mask = visibility_mask as u32;
        self
    }

    pub fn flags(mut self, flags: InstanceFlags) -> Instance<'a> {
        self.flags = flags;
        self
    }
}

pub struct InstanceArray<'i, 'a> {
    instances: &'i DeviceSlice<Instance<'a>>,
}

impl<'i, 'a> InstanceArray<'i, 'a> {
    pub fn new(instances: &'i DeviceSlice<Instance<'a>>) -> InstanceArray<'i, 'a> {
        InstanceArray { instances }
    }
}

impl Hash for InstanceArray<'_, '_> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        state.write_usize(self.instances.len());
    }
}

impl BuildInput for InstanceArray<'_, '_> {
    fn to_sys(&self) -> optix_sys::OptixBuildInput {
        let mut v = optix_sys::OptixBuildInput {
            type_: optix_sys::OptixBuildInputType::OPTIX_BUILD_INPUT_TYPE_INSTANCES,
            ..Default::default()
        };
        unsafe {
            *v.__bindgen_anon_1.instanceArray.as_mut() = optix_sys::OptixBuildInputInstanceArray {
                instances: self.instances.as_device_ptr().as_raw(),
                numInstances: self.instances.len() as u32,
                instanceStride: 0,
            };
        };
        v
    }
}

pub struct InstancePointerArray<'i> {
    instances: &'i DeviceSlice<CUdeviceptr>,
}

impl<'i> InstancePointerArray<'i> {
    pub fn new(instances: &'i DeviceSlice<CUdeviceptr>) -> InstancePointerArray<'i> {
        InstancePointerArray { instances }
    }
}

impl Hash for InstancePointerArray<'_> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        state.write_usize(self.instances.len());
    }
}

impl BuildInput for InstancePointerArray<'_> {
    fn to_sys(&self) -> optix_sys::OptixBuildInput {
        let mut v = optix_sys::OptixBuildInput {
            type_: optix_sys::OptixBuildInputType::OPTIX_BUILD_INPUT_TYPE_INSTANCE_POINTERS,
            ..Default::default()
        };
        unsafe {
            *v.__bindgen_anon_1.instanceArray.as_mut() = optix_sys::OptixBuildInputInstanceArray {
                instances: self.instances.as_device_ptr().as_raw(),
                numInstances: self.instances.len() as u32,
                instanceStride: 0,
            };
        };
        v
    }
}

/// A scene graph node holding a child node with a transform to be applied during
/// ray traversal.
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct StaticTransformWrapper(optix_sys::OptixStaticTransform);

unsafe impl DeviceCopy for StaticTransformWrapper {}

const_assert_eq!(
    std::mem::size_of::<StaticTransformWrapper>(),
    std::mem::size_of::<optix_sys::OptixStaticTransform>(),
);

/// Stores the device memory and the [`TraversableHandle`] for a [`StaticTransform`]
pub struct StaticTransform {
    #[allow(dead_code)]
    buf: DeviceBox<StaticTransformWrapper>,
    hnd: TraversableHandle,
}

impl StaticTransform {
    /// Create a new DeviceStaticTransform by copying the given [`StaticTransform`]
    /// to the device and converting the resulting pointer to an OptiX [`Traversable`];
    pub fn new<T: Traversable, M: Into<RowMatrix3x4<f32>> + Clone>(
        ctx: &DeviceContext,
        child: &T,
        transform: &M,
        inv_transform: &M,
    ) -> Result<StaticTransform> {
        let transform = (*transform).clone().into();
        let inv_transform = (*inv_transform).clone().into();
        let buf = DeviceBox::new(&StaticTransformWrapper(optix_sys::OptixStaticTransform {
            child: child.handle().inner,
            transform: transform.into(),
            invTransform: inv_transform.into(),
            ..Default::default()
        }))?;
        let hnd = unsafe {
            convert_pointer_to_traversable_handle(
                ctx,
                buf.as_device_ptr().as_raw(),
                TraversableType::StaticTransform,
            )?
        };

        Ok(StaticTransform { buf, hnd })
    }

    /// Create a new DeviceStaticTransform from device memory and pre-converted
    /// handle
    pub unsafe fn from_raw_parts(
        buf: DeviceBox<StaticTransformWrapper>,
        hnd: TraversableHandle,
    ) -> Self {
        Self { buf, hnd }
    }
}

impl Traversable for StaticTransform {
    fn handle(&self) -> TraversableHandle {
        self.hnd
    }
}

/// A scene graph node holding a child node with a motion transform to be applied
/// during ray traversal, represented as SRT Data.
///
/// Stores the device memory and the [`TraversableHandle`] for a [`optix_sys::OptixMatrixMotionTransform`]
/// and an arbitrary number of motion keys
pub struct MatrixMotionTransform {
    #[allow(dead_code)]
    buf: DeviceBuffer<u8>,
    hnd: TraversableHandle,
}

impl MatrixMotionTransform {
    /// Create a new MatrixMotionTransform with the given time range, flags and
    /// motion keys.
    ///
    /// This method handles all memory allocation and copying the data to the
    /// device.
    ///
    /// # Errors
    /// * [`Error::TooFewMotionKeys`] - If `transforms.len() < 2`
    /// * [`Error::OptixError`] - Any internal OptiX error
    /// * [`Error::CudaError`] - Any internal OptiX error
    pub fn new<T: Traversable>(
        ctx: &DeviceContext,
        child: &T,
        time_begin: f32,
        time_end: f32,
        flags: MotionFlags,
        transforms: &[RowMatrix3x4<f32>],
    ) -> Result<MatrixMotionTransform> {
        let num_keys = transforms.len();
        if num_keys < 2 {
            return Err(Error::TooFewMotionKeys(num_keys));
        }

        let mmt = optix_sys::OptixMatrixMotionTransform {
            child: child.handle().inner,
            motionOptions: optix_sys::OptixMotionOptions {
                numKeys: num_keys as u16,
                timeBegin: time_begin,
                timeEnd: time_end,
                flags: flags.bits(),
            },
            ..Default::default()
        };

        let size = size_of::<optix_sys::OptixMatrixMotionTransform>()
            + size_of::<f32>() * 12 * (num_keys - 2);

        // copy the transform data
        unsafe {
            // allocate memory for the transform struct and all the matrices
            let buf = DeviceBuffer::<u8>::uninitialized(size)?;

            // get the offset of the matrix data from the base of the struct
            let transform_ptr = buf
                .as_device_ptr()
                .add(offset_of!(optix_sys::OptixMatrixMotionTransform, transform));

            // copy the transform data.
            // Note we're writing 24 bytes of data for the transform field that
            // we'll just overwrite on the next line, but it's probably more
            // efficient to do that than to write each field individually
            cust::memory::memcpy_htod(
                buf.as_device_ptr().as_raw(),
                &mmt as *const _ as *const c_void,
                size_of::<optix_sys::OptixMatrixMotionTransform>(),
            )?;

            // copy the matrix data
            cust::memory::memcpy_htod(
                transform_ptr.as_raw(),
                transforms.as_ptr() as *const c_void,
                std::mem::size_of_val(transforms),
            )?;

            let hnd = convert_pointer_to_traversable_handle(
                ctx,
                buf.as_device_ptr().as_raw(),
                TraversableType::MatrixMotionTransform,
            )?;

            Ok(Self { buf, hnd })
        }
    }

    /// Create a new MatrixMotionTransform from device memory and pre-converted
    /// handle
    pub unsafe fn from_raw_parts(buf: DeviceBuffer<u8>, hnd: TraversableHandle) -> Self {
        Self { buf, hnd }
    }
}

impl Traversable for MatrixMotionTransform {
    fn handle(&self) -> TraversableHandle {
        self.hnd
    }
}

/// Represents an SRT transformation.
///
/// An SRT transformation can represent a smooth rotation with fewer motion keys
/// than a matrix transformation. Each motion key is constructed from elements
/// taken from a matrix $S$, a quaternion $R$, and a translation $T$.
///
/// The scaling matrix,
/// $$
/// S=\begin{bmatrix}
/// sx & a & b & pvx \cr 0 & sy & c & pvy \cr 0 & 0 & sz & pvz
/// \end{bmatrix}
/// $$
///
/// defines an affine transformation that can include scale, shear, and a translation.
/// The translation allows to define the pivot point for the subsequent rotation.
///
/// The rotation quaternion $R = [qx, qy, qz, qw]$ describes a rotation with angular
/// component $qw = \cos(\theta / 2)$ and other components
/// $[qx, qy, qz] = \sin(\theta / 2) \cdot [ax, ay, az]$ where the axis $[ax, ay, az]$
/// is normalized.
///
/// The translation matrix,
/// $$
/// T = \begin{bmatrix} 1 & 0 & 0 & tx \cr 0 & 1 & 0 & ty \cr 0 & 0 & 1 & tz \end{bmatrix}
/// $$
/// defines another translation that is applied after the rotation. Typically, this
/// translation includes the inverse translation from the matrix $S$ to reverse the
/// translation for the pivot point for $R$.
///
/// To obtain the effective transformation at time $t$, the elements of the components
/// of $S$, $R$, and $T$ will be interpolated linearly. The components are then
/// multiplied to obtain the combined transformation $C = T \cdot R \cdot S$. The
/// transformation $C$ is the effective object-to-world transformations at time $t$,
/// and $C^{-1}$ is the effective world-to-object transformation at time $t$.
///
#[repr(transparent)]
#[derive(Copy, Clone, Debug)]
pub struct SrtData(optix_sys::OptixSRTData);

unsafe impl DeviceCopy for SrtData {}

impl Deref for SrtData {
    type Target = optix_sys::OptixSRTData;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

/// A scene graph node holding a child node with a motion transform to be applied
/// during ray traversal, represented as SRT Data.
///
/// Stores the device memory and the [`TraversableHandle`] for a [`optix_sys::OptixSRTMotionTransform`]
/// and an arbitrary number of motion keys
pub struct SrtMotionTransform {
    // TODO(RDambrosio016): ask al what this is for :p
    #[allow(dead_code)]
    buf: DeviceBuffer<u8>,
    hnd: TraversableHandle,
}

impl SrtMotionTransform {
    /// Create a new SrtMotionTransform from the given child [`TraversableHandle`],
    /// time range, flags and [`SrtData`]
    ///
    /// This method handles all memory allocation and copying the data to the
    /// device.
    ///
    /// # Errors
    /// * [`Error::TooFewMotionKeys`] - If `srt_data.len() < 2`
    /// * [`Error::OptixError`] - Any internal OptiX error
    /// * [`Error::CudaError`] - Any internal OptiX error
    pub fn new<T: Traversable>(
        ctx: &DeviceContext,
        child: &T,
        time_begin: f32,
        time_end: f32,
        flags: MotionFlags,
        srt_data: &[SrtData],
    ) -> Result<SrtMotionTransform> {
        let num_keys = srt_data.len();
        if num_keys < 2 {
            return Err(Error::TooFewMotionKeys(num_keys));
        }

        let mmt = optix_sys::OptixSRTMotionTransform {
            child: child.handle().inner,
            motionOptions: optix_sys::OptixMotionOptions {
                numKeys: num_keys as u16,
                timeBegin: time_begin,
                timeEnd: time_end,
                flags: flags.bits(),
            },
            ..Default::default()
        };

        let size = size_of::<optix_sys::OptixSRTMotionTransform>()
            + size_of::<f32>() * size_of::<SrtData>() * (num_keys - 2);

        // copy the transform data
        unsafe {
            // allocate memory for the transform struct and all the matrices
            let buf = DeviceBuffer::<u8>::uninitialized(size)?;

            // get the offset of the matrix data from the base of the struct
            let transform_ptr = buf
                .as_device_ptr()
                .add(offset_of!(optix_sys::OptixSRTMotionTransform, srtData));

            // copy the transform data.
            // Note we're writing 24 bytes of data for the transform field that
            // we'll just overwrite on the next line, but it's probably more
            // efficient to do that than to write each field individually
            cust::memory::memcpy_htod(
                buf.as_device_ptr().as_raw(),
                &mmt as *const _ as *const c_void,
                size_of::<optix_sys::OptixSRTMotionTransform>(),
            )?;

            // copy the matrix data
            cust::memory::memcpy_htod(
                transform_ptr.as_raw(),
                srt_data.as_ptr() as *const c_void,
                std::mem::size_of_val(srt_data),
            )?;

            let hnd = convert_pointer_to_traversable_handle(
                ctx,
                buf.as_device_ptr().as_raw(),
                TraversableType::SrtMotionTransform,
            )?;

            Ok(Self { buf, hnd })
        }
    }

    /// Create a new SrtMotionTransform from device memory and pre-converted
    /// handle
    pub unsafe fn from_raw_parts(buf: DeviceBuffer<u8>, hnd: TraversableHandle) -> Self {
        Self { buf, hnd }
    }
}

impl Traversable for SrtMotionTransform {
    fn handle(&self) -> TraversableHandle {
        self.hnd
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TraversableType {
    StaticTransform,
    MatrixMotionTransform,
    SrtMotionTransform,
}

impl From<TraversableType> for optix_sys::OptixTraversableType {
    fn from(t: TraversableType) -> Self {
        match t {
            TraversableType::StaticTransform => {
                optix_sys::OptixTraversableType::OPTIX_TRAVERSABLE_TYPE_STATIC_TRANSFORM
            }
            TraversableType::MatrixMotionTransform => {
                optix_sys::OptixTraversableType::OPTIX_TRAVERSABLE_TYPE_MATRIX_MOTION_TRANSFORM
            }
            TraversableType::SrtMotionTransform => {
                optix_sys::OptixTraversableType::OPTIX_TRAVERSABLE_TYPE_SRT_MOTION_TRANSFORM
            }
        }
    }
}

/// Convert a device pointer into a [`TraversableHandle`].
///
/// OptiX transform traversables are managed by the application. Once you have
/// created your transform and copied it to the device, use this to get a
/// [`TraversableHandle`] from it.
pub unsafe fn convert_pointer_to_traversable_handle(
    ctx: &DeviceContext,
    ptr: CUdeviceptr,
    pointer_type: TraversableType,
) -> Result<TraversableHandle> {
    let mut inner = 0;
    Ok(optix_call!(optixConvertPointerToTraversableHandle(
        ctx.raw,
        ptr,
        pointer_type.into(),
        &mut inner
    ))
    .map(|_| TraversableHandle { inner })?)
}
