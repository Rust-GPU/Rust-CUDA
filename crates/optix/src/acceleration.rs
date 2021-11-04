//! # Acceleration Structures
//!
//! NVIDIA OptiX 7 provides acceleration structures to optimize the search for the
//! intersection of rays with the geometric data in the scene. Acceleration structures
//! can contain two types of data: geometric primitives (a geometry-AS) or instances
//! (an instance-AS). Acceleration structures are created on the device using a set
//! of functions. These functions enable overlapping and pipelining of acceleration
//! structure creation, called a build. The functions use one or more [`BuildInput`]
//! structs to specify the geometry plus a set of parameters to control the build.
//!
//! Acceleration structures have size limits, listed in “Limits”. For an instance
//! acceleration structure, the number of instances has an upper limit. For a geometry
//! acceleration structure, the number of geometric primitives is limited,
//! specifically the total number of primitives in its build inputs, multiplied by the
//! number of motion keys.
//!  
//! The following acceleration structure types are supported:
//!
//! #### Instance acceleration structures
//! - [`InstanceArray`](crate::instance_array::InstanceArray)
//! - [`InstancePointerArray`](crate::instance_array::InstancePointerArray)
//!
//! #### Geometry acceleration structure containing built-in triangles
//! - [`TriangleArray`](crate::triangle_array::TriangleArray)
//! - [`IndexedTriangleArray`](crate::triangle_array::IndexedTriangleArray)
//!
//! #### Geometry acceleration structure containing built-in curves
//! - [`CurveArray`](crate::curve_array::CurveArray)
//!
//! #### Geometry acceleration structure containing custom primitives
//! - [`CustomPrimitiveArray`](crate::custom_primitive_array::CustomPrimitiveArray)
//!
//! ## Building
//!
//! For geometry-AS builds, each build input can specify a set of triangles, a set
//! of curves, or a set of user-defined primitives bounded by specified axis-aligned
//! bounding boxes. Multiple build inputs can be passed as an array to [`accel_build`]
//! to combine different meshes into a single acceleration structure. All build
//! inputs for a single build must agree on the build input type.
//!
//! Instance acceleration structures have a single build input and specify an array
//! of instances. Each [`Instance`](crate::instance_array::Instance) includes a ray transformation and an
//! [`TraversableHandle`] that refers to a geometry-AS, a transform node, or another
//! instance acceleration structure.
//!
//! ### Safe API
//!
//! The easiest way to build an acceleration structure is using [`Accel::build`]
//! to which you just pass a slice of [`BuildInput`]s and the function handles
//! memory allocation and synchronization for you.
//!
//! This is handy for getting something working with the minimum of fuss, but
//! means reallocating temporary storage each time. It also means synchronizing
//! after each build rather than potentially processing many builds on a stream
//! and synchronizing at the end.
//!
//! ```no_run
//! use cust::prelude as cu;
//! use optix::prelude as ox;
//! # fn doit() -> Result<(), Box<dyn std::error::Error>> {
//! # cust::init(cu::CudaFlags::empty())?;
//! # ox::init()?;
//! # let device = cu::Device::get_device(0)?;
//! # let cu_ctx = cu::Context::create_and_push(cu::ContextFlags::SCHED_AUTO |
//! # cu::ContextFlags::MAP_HOST, device)?;
//! # let ctx = ox::DeviceContext::new(&cu_ctx, false)?;
//! # let vertices: Vec<[f32; 3]> = Vec::new();
//! # let indices: Vec<[u32; 3]> = Vec::new();
//! # let stream = cu::Stream::new(cu::StreamFlags::DEFAULT, None)?;
//!
//! let buf_vertex = cu::DeviceBuffer::from_slice(&vertices)?;
//! let buf_indices = cu::DeviceBuffer::from_slice(&indices)?;
//!
//! let geometry_flags = ox::GeometryFlags::None;
//! let triangle_input =
//!     ox::IndexedTriangleArray::new(
//!         &[&buf_vertex],
//!         &buf_indices,
//!         &[geometry_flags]
//!     );
//!
//! let accel_options =
//!     ox::AccelBuildOptions::new(
//!         ox::BuildFlags::ALLOW_COMPACTION,
//!         ox::BuildOperation::Build
//!     );
//!
//! let build_inputs = vec![triangle_input];
//!
//! let gas = ox::Accel::build(
//!     &ctx,
//!     &stream,
//!     &[accel_options],
//!     &build_inputs,
//!     true
//! )?;
//!
//! stream.synchronize()?;
//! # Ok(())
//! # }
//! ```
//!
//! ### Unsafe API
//!
//! As an alternative, you can also use the unsafe functions [`accel_build`],
//! [`accel_compact`], and [`Accel::from_raw_parts`] to handle the memory
//! allocation yourself, meaning you can reuse buffers between accel builds.
//!
//! To prepare for a build, the required memory sizes are queried by passing an
//! initial set of build inputs and parameters to [`accel_compute_memory_usage`].
//! It returns three different sizes:
//!
//! * `output_size_in_bytes` - Size of the memory region where the resulting
//! acceleration structure is placed. This size is an upper bound and may be
//! substantially larger than the final acceleration structure. (See “Compacting acceleration structures”.)
//! * `temp_size_in_bytes` - Size of the memory region that is temporarily used during
//! the build.
//! * `temp_update_size_in_bytes` - Size of the memory region that is temporarily
//! required to update the acceleration structure.
//!
//! Using these sizes, the application allocates memory for the output and temporary
//! memory buffers on the device. The pointers to these buffers must be aligned to
//! a 128-byte boundary. These buffers are actively used for the duration of the
//! build. For this reason, they cannot be shared with other currently active build
//! requests.
//!
//! Note that [`accel_compute_memory_usage`] does not initiate any activity on the
//! device; pointers to device memory or contents of input buffers are not required to point to allocated memory.
//!
//! The function [`accel_build`] takes the same array of [`BuildInput`] structs as
//! [`accel_compute_memory_usage`] and builds a single acceleration structure from
//! these inputs. This acceleration structure can contain either geometry or
//! instances, depending on the inputs to the build.
//!
//! The build operation is executed on the device in the specified CUDA stream and
//! runs asynchronously on the device, similar to CUDA kernel launches. The
//! application may choose to block the host-side thread or synchronize with other
//! CUDA streams by using available CUDA synchronization functionality such as
//! [`Stream::synchronize()`](cust::stream::Stream::synchronize) or CUDA events.
//! The traversable handle returned is computed on the host and is returned from
//! the function immediately, without waiting for the build to finish. By producing
//! handles at acceleration time, custom handles can also be generated based on
//! input to the builder.
//!
//! The acceleration structure constructed by [`accel_build`] does not reference
//! any of the device buffers referenced in the build inputs. All relevant data
//! is copied from these buffers into the acceleration output buffer, possibly in
//! a different format.
//!
//! The application is free to release this memory after the build without
//! invalidating the acceleration structure. However, instance-AS builds will
//! continue to refer to other instance-AS and geometry-AS instances and transform
//! nodes.
//!
//! ```no_run
//! use cust::prelude as cu;
//! use optix::prelude as ox;
//! # fn doit() -> Result<(), Box<dyn std::error::Error>> {
//! # cust::init(cu::CudaFlags::empty())?;
//! # ox::init()?;
//! # let device = cu::Device::get_device(0)?;
//! # let cu_ctx = cu::Context::create_and_push(cu::ContextFlags::SCHED_AUTO |
//! # cu::ContextFlags::MAP_HOST, device)?;
//! # let ctx = ox::DeviceContext::new(&cu_ctx, false)?;
//! # let vertices: Vec<[f32; 3]> = Vec::new();
//! # let indices: Vec<[u32; 3]> = Vec::new();
//! # let stream = cu::Stream::new(cu::StreamFlags::DEFAULT, None)?;
//!
//! let buf_vertex = cu::DeviceBuffer::from_slice(&vertices)?;
//! let buf_indices = cu::DeviceBuffer::from_slice(&indices)?;
//!
//! let geometry_flags = ox::GeometryFlags::None;
//!
//! let build_inputs =
//!     [ox::IndexedTriangleArray::new(
//!         &[&buf_vertex],
//!         &buf_indices,
//!         &[geometry_flags]
//!     )];
//!
//! let accel_options =
//!     ox::AccelBuildOptions::new(
//!         ox::BuildFlags::ALLOW_COMPACTION,
//!         ox::BuildOperation::Build
//!     );
//!
//! // Get the storage requirements for temporary and output buffers
//! let sizes = accel_compute_memory_usage(ctx, accel_options, build_inputs)?;
//!
//! // Allocate temporary and output buffers
//! let mut output_buffer =
//!     unsafe { DeviceBuffer::<u8>::uninitialized(sizes.output_size_in_bytes)? };
//! let mut temp_buffer =
//!     unsafe { DeviceBuffer::<u8>::uninitialized(sizes.temp_size_in_bytes)? };
//!
//! // Build the accel
//! let hnd = unsafe {
//!     accel_build(
//!         ctx,
//!         stream,
//!         accel_options,
//!         build_inputs,
//!         &mut temp_buffer,
//!         &mut output_buffer,
//!         &mut properties,
//!     )?
//! };
//!
//! // The accel build is asynchronous
//! stream.synchronize()?;
//!
//! # Ok(())
//! # }
//! ```
//!
//! ## Primitive Build Inputs
//! The [`accel_build`] function accepts multiple build inputs per call, but they
//! must be all triangle inputs, all curve inputs, or all AABB inputs. Mixing build
//! input types in a single geometry-AS is not allowed.
//!
//! Each build input maps to one or more consecutive records in the shader binding
//! table (SBT), which controls program dispatch. (See “Shader binding table”.) If
//! multiple records in the SBT are required, the application needs to provide a
//! device buffer with per-primitive SBT record indices for that build input. If
//! only a single SBT record is requested, all primitives reference this same unique
//! SBT record. Note that there is a limit to the number of referenced SBT records
//! per geometry-AS. (Limits are discussed in “Limits”.)
//!
//! Each build input also specifies an array of OptixGeometryFlags, one for each SBT
//! record. The flags for one record apply to all primitives mapped to this SBT record.
//!
//! The following flags are supported:
//!
//! * [`GeometryFlags::None`] - Applies the default behavior when calling the any-hit
//! program, possibly multiple times, allowing the acceleration-structure builder
//! to apply all optimizations.
//! * [`GeometryFlags::RequireSingleAnyHitCall`] - Disables some optimizations
//! specific to acceleration-structure builders. By default, traversal may call
//! the any-hit program more than once for each intersected primitive. Setting
//! the flag ensures that the any-hit program is called only once for a hit with a
//! primitive. However, setting this flag may change traversal performance. The
//! usage of this flag may be required for correctness of some rendering algorithms;
//! for example, in cases where opacity or transparency information is accumulated
//! in an any-hit program.
//! * [`GeometryFlags::DisableAnyHit`] - Indicates that traversal should not call
//! the any-hit program for this primitive even if the corresponding SBT record
//! contains an any-hit program. Setting this flag usually improves performance
//! even if no any-hit program is present in the SBT.
//!
//! Primitives inside a build input are indexed starting from zero. This primitive
//! index is accessible inside the intersection, any-hit, and closest-hit programs.
//! If the application chooses to offset this index for all primitives in a build
//! input, there is no overhead at runtime. This can be particularly useful when
//! data for consecutive build inputs is stored consecutively in device memory.
//! The `primitive_index_offset` value is only used when reporting the intersection
//! primitive.
//!
//! ## Build Flags
//!
//! An acceleration structure build can be controlled using the values of the
//! [`BuildFlags`] enum. To enable random vertex access on an acceleration structure,
//! use [`BuildFlags::ALLOW_RANDOM_VERTEX_ACCESS`]. (See “Vertex random access”.)
//! To steer trade-offs between build performance, runtime traversal performance
//! and acceleration structure memory usage, use [`BuildFlags::PREFER_FAST_TRACE`]
//! and [`BuildFlags::PREFER_FAST_BUILD`]. For curve primitives in particular,
//! these flags control splitting; see “Splitting curve segments”.
//!
//! The flags [`BuildFlags::PREFER_FAST_TRACE`] and [`BuildFlags::PREFER_FAST_BUILD`]
//! are mutually exclusive. To combine multiple flags that are not mutually exclusive,
//! use the logical “or” operator.
//!
//! ## Dynamic Updates
//!
//! Building an acceleration structure can be computationally costly. Applications
//! may choose to update an existing acceleration structure using modified vertex
//! data or bounding boxes. Updating an existing acceleration structure is generally
//! much faster than rebuilding. However, the quality of the acceleration structure
//! may degrade if the data changes too much with an update, for example, through
//! explosions or other chaotic transitions—even if for only parts of the mesh.
//! The degraded acceleration structure may result in slower traversal performance
//! as compared to an acceleration structure built from scratch from the modified
//! input data.
//!
//! ### Safe API
//!
//! The simplest way to use dynamic updates is with the [`DynamicAccel`] structure.
//! Simply call [`DynamicAccel::new()`] as you would with [`Accel`], and then
//! call [`DynamicAccel::update()`] with the updated build inputs when you want
//! to update the acceleration structure.
//!
//! Note that the inputs to [`DynamicAccel::update`] must have the same structure,
//! i.e. the number of motion keys, aabbs, triangle topology etc must be the same,
//! although the underlying data (including the data pointers) can be different.
//! If the data have a different structure, then behaviour is undefined.
//! [`DynamicAccel`] checks this by hashing the inputs and returns an error if
//! the data do not match.
//!
//! ### Unsafe API
//!
//! To allow for future updates of an acceleration structure, set
//! [`BuildFlags::ALLOW_UPDATE`] in the build flags when building the acceleration
//! structure initially.
//!
//! To update the previously built acceleration structure, set the operation to
//! [`BuildOperation::Update`] and then call [`accel_build()`] on the same output
//! data. All other options are required to be identical to the original build.
//! The update is done in-place on the output data.
//!
//! Updating an acceleration structure usually requires a different amount of temporary memory than the original build.
//!
//! When updating an existing acceleration structure, only the device pointers and/or
//! their buffer content may be changed. You cannot change the number of build inputs,
//! the build input types, build flags, traversable handles for instances (for an
//! instance-AS), or the number of vertices, indices, AABBs, instances, SBT records
//! or motion keys. Changes to any of these things may result in undefined behavior,
//! including GPU faults.
//!
//! Note the following:
//!
//! * When using indices, changing the connectivity or, in general, using shuffled
//! vertex positions will work, but the quality of the acceleration structure will
//! likely degrade substantially.
//! * During an animation operation, geometry that should be invisible to the camera
//! should not be “removed” from the scene, either by moving it very far away or
//! by converting it into a degenerate form. Such changes to the geometry will also
//! degrade the acceleration structure.
//! * In these cases, it is more efficient to re-build the geometry-AS and/or the
//! instance-AS, or to use the respective masking and flags.
//!
//! Updating an acceleration structure requires that any other acceleration structure
//! that is using this acceleration structure as a child directly or indirectly
//! also needs to be updated or rebuild.

use crate::{context::DeviceContext, error::Error, optix_call, sys};
use cust::{
    memory::{CopyDestination, DeviceBox, DeviceBuffer, DevicePointer, DeviceSlice},
    DeviceCopy,
};
type Result<T, E = Error> = std::result::Result<T, E>;

use std::ops::Deref;
use std::{
    collections::hash_map::DefaultHasher,
    hash::{Hash, Hasher},
};

pub trait BuildInput: std::hash::Hash {
    fn to_sys(&self) -> sys::OptixBuildInput;
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
    buf: DeviceBuffer<u8>,
    hnd: TraversableHandle,
}

impl Accel {
    /// Get the [`TraversableHandle`] that represents this accel.
    pub fn handle(&self) -> TraversableHandle {
        self.hnd
    }

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

        let mut compacted_size_buffer = unsafe { DeviceBox::<usize>::uninitialized()? };

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

            let mut buf = unsafe { DeviceBuffer::<u8>::uninitialized(compacted_size)? };

            let hnd = unsafe { accel_compact(ctx, stream, hnd, &mut buf)? };

            Ok(Accel { buf, hnd })
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
        let mut inner = sys::OptixAccelRelocationInfo::default();
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

        let mut compacted_size_buffer = unsafe { DeviceBox::<usize>::uninitialized()? };

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

        let mut compacted_size_buffer = unsafe { DeviceBox::<usize>::uninitialized()? };

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
#[derive(Copy, Clone, Debug, PartialEq, DeviceCopy)]
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
    let properties: Vec<sys::OptixAccelEmitDesc> =
        emitted_properties.iter_mut().map(|p| p.into()).collect();

    let build_sys: Vec<_> = build_inputs.iter().map(|b| b.to_sys()).collect();

    Ok(optix_call!(optixAccelBuild(
        ctx.raw,
        stream.as_inner(),
        accel_options.as_ptr() as *const _,
        build_sys.as_ptr(),
        build_sys.len() as u32,
        temp_buffer.as_device_ptr(),
        temp_buffer.len(),
        output_buffer.as_device_ptr(),
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
        output_buffer.as_device_ptr(),
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
    pub struct BuildFlags: u32 {
        const NONE = sys::OptixBuildFlags_OPTIX_BUILD_FLAG_NONE;
        const ALLOW_UPDATE = sys::OptixBuildFlags_OPTIX_BUILD_FLAG_ALLOW_UPDATE;
        const ALLOW_COMPACTION = sys::OptixBuildFlags_OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
        const PREFER_FAST_TRACE = sys::OptixBuildFlags_OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
        const PREFER_FAST_BUILD = sys::OptixBuildFlags_OPTIX_BUILD_FLAG_PREFER_FAST_BUILD;
        const ALLOW_RANDOM_VERTEX_ACCESS = sys::OptixBuildFlags_OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS;
    }
}

/// Select which operation to perform with [`accel_build()`].
#[repr(u32)]
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum BuildOperation {
    Build = sys::OptixBuildOperation_OPTIX_BUILD_OPERATION_BUILD,
    Update = sys::OptixBuildOperation_OPTIX_BUILD_OPERATION_UPDATE,
}

bitflags::bitflags! {
    /// Configure how to handle ray times that are outside of the provided motion keys.
    ///
    /// By default, the object will appear static (clamped) to the nearest motion
    /// key for rays outside of the range of key times.
    ///
    /// * `START_VANISH` - The object will be invisible to rays with a time less
    /// than the first provided motion key
    /// * `END_VANISH` - The object will be invisible to rays with a time less
    /// than the first provided motion key
    pub struct MotionFlags: u16 {
        const NONE = sys::OptixMotionFlags_OPTIX_MOTION_FLAG_NONE as u16;
        const START_VANISH = sys::OptixMotionFlags_OPTIX_MOTION_FLAG_START_VANISH as u16;
        const END_VANISH = sys::OptixMotionFlags_OPTIX_MOTION_FLAG_END_VANISH as u16;
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
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct MotionOptions {
    pub num_keys: u16,
    pub flags: MotionFlags,
    pub time_begin: f32,
    pub time_end: f32,
}

/// Options to configure the [`accel_build()`]
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct AccelBuildOptions {
    build_flags: BuildFlags,
    operation: BuildOperation,
    motion_options: MotionOptions,
}

impl AccelBuildOptions {
    /// Create a new AccelBuildOptions with the given flags and operation and
    /// no motion blur.
    pub fn new(build_flags: BuildFlags, operation: BuildOperation) -> Self {
        AccelBuildOptions {
            build_flags,
            operation,
            motion_options: MotionOptions {
                num_keys: 1,
                flags: MotionFlags::NONE,
                time_begin: 0.0f32,
                time_end: 1.0f32,
            },
        }
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
    inner: sys::OptixAccelRelocationInfo,
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
/// // Copy the returned size needed for the compacted buffer and allocate
/// // storage
/// let mut compacted_size = 0usize;
/// compacted_size_buffer.copy_to(&mut compacted_size)?;
///
/// let mut buf = unsafe { DeviceBuffer::<u8>::uninitialized(compacted_size)? };
///
/// // Compact the accel structure.
/// let hnd = unsafe { accel_compact(ctx, stream, hnd, &mut buf)? };

pub enum AccelEmitDesc {
    CompactedSize(DevicePointer<usize>),
    Aabbs(DevicePointer<Aabb>),
}

/// Struct representing a bounding box.
#[repr(C)]
#[derive(DeviceCopy, Copy, Clone)]
pub struct Aabb {
    min_x: f32,
    min_y: f32,
    min_z: f32,
    max_x: f32,
    max_y: f32,
    max_z: f32,
}

impl From<&mut AccelEmitDesc> for sys::OptixAccelEmitDesc {
    fn from(aed: &mut AccelEmitDesc) -> Self {
        match aed {
            AccelEmitDesc::CompactedSize(p) => Self {
                result: p.as_raw(),
                type_: sys::OptixAccelPropertyType_OPTIX_PROPERTY_TYPE_COMPACTED_SIZE,
            },
            AccelEmitDesc::Aabbs(p) => Self {
                result: p.as_raw(),
                type_: sys::OptixAccelPropertyType_OPTIX_PROPERTY_TYPE_AABBS,
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
    None = sys::OptixGeometryFlags::None as u32,
    DisableAnyHit = sys::OptixGeometryFlags::DisableAnyHit as u32,
    RequireSingleAnyHitCall = sys::OptixGeometryFlags::RequireSingleAnyHitCall as u32,
}

impl From<GeometryFlags> for sys::OptixGeometryFlags {
    fn from(f: GeometryFlags) -> Self {
        match f {
            GeometryFlags::None => sys::OptixGeometryFlags::None,
            GeometryFlags::DisableAnyHit => sys::OptixGeometryFlags::DisableAnyHit,
            GeometryFlags::RequireSingleAnyHitCall => {
                sys::OptixGeometryFlags::RequireSingleAnyHitCall
            }
        }
    }
}

impl From<GeometryFlags> for u32 {
    fn from(f: GeometryFlags) -> Self {
        match f {
            GeometryFlags::None => sys::OptixGeometryFlags::None as u32,
            GeometryFlags::DisableAnyHit => sys::OptixGeometryFlags::DisableAnyHit as u32,
            GeometryFlags::RequireSingleAnyHitCall => {
                sys::OptixGeometryFlags::RequireSingleAnyHitCall as u32
            }
        }
    }
}
