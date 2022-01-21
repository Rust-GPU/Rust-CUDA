# Acceleration Structures

```no_run
use cust::prelude as cu;
use optix::prelude as ox;
# fn doit() -> Result<(), Box<dyn std::error::Error>> {
# cust::init(cu::CudaFlags::empty())?;
# ox::init()?;
# let device = cu::Device::get_device(0)?;
# let cu_ctx = cu::Context::create_and_push(cu::ContextFlags::SCHED_AUTO |
# cu::ContextFlags::MAP_HOST, device)?;
# let ctx = ox::DeviceContext::new(&cu_ctx, false)?;
# let vertices: Vec<[f32; 3]> = Vec::new();
# let indices: Vec<[u32; 3]> = Vec::new();
# let stream = cu::Stream::new(cu::StreamFlags::DEFAULT, None)?;

// Allocate buffers and copy vertex and index data to device
let buf_vertex = cu::DeviceBuffer::from_slice(&vertices)?;
let buf_indices = cu::DeviceBuffer::from_slice(&indices)?;

// Tell OptiX the structure of our triangle mesh
let geometry_flags = ox::GeometryFlags::None;
let triangle_input =
    ox::IndexedTriangleArray::new(
        &[&buf_vertex],
        &buf_indices,
        &[geometry_flags]
    );

// Tell OptiX we'd prefer a faster traversal over a faster bvh build.
let accel_options = AccelBuildOptions::new(ox::BuildFlags::PREFER_FAST_TRACE);

// Build the accel asynchronously
let gas = ox::Accel::build(
    &ctx,
    &stream,
    &[accel_options],
    &[triangle_input],
    true
)?;
# Ok(())
# }
```

# Programming Guide...
<details>
<summary>Click here to expand programming guide</summary>

## Contents

- [Building](#building)
    - [Building Safe API](#building-safe-api)
    - [Buliding Unsafe API](#building-unsafe-api)
- [Primitive Build Inputs](#primitive-build-inputs)
- [Build Flags](#build-flags)
- [Dynamic Updates](#dynamic-updates)
    - [Dynamic Updates Safe API](#dynamic-updates-safe-api)
    - [Dynamic Updates Unsafe API](#dynamic-updates-unsafe-api)
- [Compaction](#compaction)
    - [Compaction Safe API](#compaction-safe-api)
    - [Compaction Unsafe API](#compaction-unsafe-api)
- [Traversable Objects](#traversable-objects)
    - [Traversable Objects Safe API](#traversable-objects-safe-api)
    - [Traversable Objects Unsafe API](#traversable-objects-unsafe-api)
- [Motion Blur](#motion-blur)
    - [Basics](#basics)
    - [Motion Geometry Acceleration Structure](#motion-geometry-acceleration-structure)
    - [Motion Instance Acceleration Structure](#motion-instance-acceleration-structure)
    - [Motion Matrix Transform](#motion-matrix-transform)
    - [Motion Scale Rotate Translate Transform](#motion-scale-rotate-translate-transform)
    - [Transforms Trade-Offs](#transforms-trade-offs)


NVIDIA OptiX 7 provides acceleration structures to optimize the search for the
intersection of rays with the geometric data in the scene. Acceleration structures
can contain two types of data: geometric primitives (a geometry-AS) or instances
(an instance-AS). Acceleration structures are created on the device using a set
of functions. These functions enable overlapping and pipelining of acceleration
structure creation, called a build. The functions use one or more [`BuildInput`]
structs to specify the geometry plus a set of parameters to control the build.

Acceleration structures have size limits, listed in “Limits”. For an instance
acceleration structure, the number of instances has an upper limit. For a geometry
acceleration structure, the number of geometric primitives is limited,
specifically the total number of primitives in its build inputs, multiplied by the
number of motion keys.
 
The following acceleration structure types are supported:

#### Instance acceleration structures
- [`InstanceArray`](crate::instance_array::InstanceArray)
- [`InstancePointerArray`](crate::instance_array::InstancePointerArray)

#### Geometry acceleration structure containing built-in triangles
- [`TriangleArray`](crate::triangle_array::TriangleArray)
- [`IndexedTriangleArray`](crate::triangle_array::IndexedTriangleArray)

#### Geometry acceleration structure containing built-in curves
- [`CurveArray`](crate::curve_array::CurveArray)

#### Geometry acceleration structure containing custom primitives
- [`CustomPrimitiveArray`](crate::custom_primitive_array::CustomPrimitiveArray)

## Building

For geometry-AS builds, each build input can specify a set of triangles, a set
of curves, or a set of user-defined primitives bounded by specified axis-aligned
bounding boxes. Multiple build inputs can be passed as an array to [`Accel::build()`]
to combine different meshes into a single acceleration structure. All build
inputs for a single build must agree on the build input type.

Instance acceleration structures have a single build input and specify an array
of instances. Each [`Instance`] includes a ray transformation and a
[`TraversableHandle`] that refers to a geometry-AS, a transform node, or another
instance acceleration structure.

### Building Safe API

The easiest way to build an acceleration structure is using [`Accel::build()`]
to which you just pass a slice of [`BuildInput`]s and the function handles
memory allocation and synchronization for you.

This is handy for getting something working with the minimum of fuss, but
means reallocating temporary storage each time. It also means synchronizing
after each build rather than potentially processing many builds on a stream
and synchronizing at the end.

```no_run
use cust::prelude as cu;
use optix::prelude as ox;
# fn doit() -> Result<(), Box<dyn std::error::Error>> {
# cust::init(cu::CudaFlags::empty())?;
# ox::init()?;
# let device = cu::Device::get_device(0)?;
# let cu_ctx = cu::Context::create_and_push(cu::ContextFlags::SCHED_AUTO |
# cu::ContextFlags::MAP_HOST, device)?;
# let ctx = ox::DeviceContext::new(&cu_ctx, false)?;
# let vertices: Vec<[f32; 3]> = Vec::new();
# let indices: Vec<[u32; 3]> = Vec::new();
# let stream = cu::Stream::new(cu::StreamFlags::DEFAULT, None)?;

let buf_vertex = cu::DeviceBuffer::from_slice(&vertices)?;
let buf_indices = cu::DeviceBuffer::from_slice(&indices)?;

let geometry_flags = ox::GeometryFlags::None;
let triangle_input =
    ox::IndexedTriangleArray::new(
        &[&buf_vertex],
        &buf_indices,
        &[geometry_flags]
    );

let accel_options =
    ox::AccelBuildOptions::new(
        ox::BuildFlags::ALLOW_COMPACTION,
        ox::BuildOperation::Build
    );

let build_inputs = vec![triangle_input];

let gas = ox::Accel::build(
    &ctx,
    &stream,
    &[accel_options],
    &build_inputs,
    true
)?;

stream.synchronize()?;
# Ok(())
# }
```

### Building Unsafe API

As an alternative, you can also use the unsafe functions [`accel_build()`],
[`accel_compact()`], and [`Accel::from_raw_parts()`] to handle the memory
allocation yourself, meaning you can reuse buffers between accel builds.

To prepare for a build, the required memory sizes are queried by passing an
initial set of build inputs and parameters to [`accel_compute_memory_usage()`].
It returns three different sizes:

* `output_size_in_bytes` - Size of the memory region where the resulting
acceleration structure is placed. This size is an upper bound and may be
substantially larger than the final acceleration structure. (See “Compacting acceleration structures”.)
* `temp_size_in_bytes` - Size of the memory region that is temporarily used during
the build.
* `temp_update_size_in_bytes` - Size of the memory region that is temporarily
required to update the acceleration structure.

Using these sizes, the application allocates memory for the output and temporary
memory buffers on the device. The pointers to these buffers must be aligned to
a 128-byte boundary. These buffers are actively used for the duration of the
build. For this reason, they cannot be shared with other currently active build
requests.

Note that [`accel_compute_memory_usage()`] does not initiate any activity on the
device; pointers to device memory or contents of input buffers are not required to point to allocated memory.

The function [`accel_build()`] takes the same array of [`BuildInput`] structs as
[`accel_compute_memory_usage()`] and builds a single acceleration structure from
these inputs. This acceleration structure can contain either geometry or
instances, depending on the inputs to the build.

The build operation is executed on the device in the specified CUDA stream and
runs asynchronously on the device, similar to CUDA kernel launches. The
application may choose to block the host-side thread or synchronize with other
CUDA streams by using available CUDA synchronization functionality such as
[`Stream::synchronize()`](cust::stream::Stream::synchronize) or CUDA events.
The traversable handle returned is computed on the host and is returned from
the function immediately, without waiting for the build to finish. By producing
handles at acceleration time, custom handles can also be generated based on
input to the builder.

The acceleration structure constructed by [`accel_build()`] does not reference
any of the device buffers referenced in the build inputs. All relevant data
is copied from these buffers into the acceleration output buffer, possibly in
a different format.

The application is free to release this memory after the build without
invalidating the acceleration structure. However, instance-AS builds will
continue to refer to other instance-AS and geometry-AS instances and transform
nodes.

```no_run
use cust::prelude as cu;
use optix::prelude as ox;
# fn doit() -> Result<(), Box<dyn std::error::Error>> {
# cust::init(cu::CudaFlags::empty())?;
# ox::init()?;
# let device = cu::Device::get_device(0)?;
# let cu_ctx = cu::Context::create_and_push(cu::ContextFlags::SCHED_AUTO |
# cu::ContextFlags::MAP_HOST, device)?;
# let ctx = ox::DeviceContext::new(&cu_ctx, false)?;
# let vertices: Vec<[f32; 3]> = Vec::new();
# let indices: Vec<[u32; 3]> = Vec::new();
# let stream = cu::Stream::new(cu::StreamFlags::DEFAULT, None)?;

let buf_vertex = cu::DeviceBuffer::from_slice(&vertices)?;
let buf_indices = cu::DeviceBuffer::from_slice(&indices)?;

let geometry_flags = ox::GeometryFlags::None;

let build_inputs =
    [ox::IndexedTriangleArray::new(
        &[&buf_vertex],
        &buf_indices,
        &[geometry_flags]
    )];

let accel_options =
    ox::AccelBuildOptions::new(
        ox::BuildFlags::ALLOW_COMPACTION,
        ox::BuildOperation::Build
    );

// Get the storage requirements for temporary and output buffers
let sizes = accel_compute_memory_usage(ctx, accel_options, build_inputs)?;

// Allocate temporary and output buffers
let mut output_buffer =
    unsafe { DeviceBuffer::<u8>::uninitialized(sizes.output_size_in_bytes)? };
let mut temp_buffer =
    unsafe { DeviceBuffer::<u8>::uninitialized(sizes.temp_size_in_bytes)? };

// Build the accel
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

// The accel build is asynchronous
stream.synchronize()?;

# Ok(())
# }
```

## Primitive Build Inputs
The [`accel_build`] function accepts multiple build inputs per call, but they
must be all triangle inputs, all curve inputs, or all AABB inputs. Mixing build
input types in a single geometry-AS is not allowed.

Each build input maps to one or more consecutive records in the shader binding
table (SBT), which controls program dispatch. (See [Shader binding table](crate::shader_binding_table).) If
multiple records in the SBT are required, the application needs to provide a
device buffer with per-primitive SBT record indices for that build input. If
only a single SBT record is requested, all primitives reference this same unique
SBT record. Note that there is a limit to the number of referenced SBT records
per geometry-AS. (Limits are discussed in “Limits”.)

Each build input also specifies an array of [`GeometryFlags`], one for each SBT
record. The flags for one record apply to all primitives mapped to this SBT record.

The following flags are supported:

* [`GeometryFlags::None`](crate::acceleration::GeometryFlags) - Applies the default behavior when calling the any-hit
program, possibly multiple times, allowing the acceleration-structure builder
to apply all optimizations.
* [`GeometryFlags::RequireSingleAnyHitCall`](crate::acceleration::GeometryFlags) - Disables some optimizations
specific to acceleration-structure builders. By default, traversal may call
the any-hit program more than once for each intersected primitive. Setting
the flag ensures that the any-hit program is called only once for a hit with a
primitive. However, setting this flag may change traversal performance. The
usage of this flag may be required for correctness of some rendering algorithms;
for example, in cases where opacity or transparency information is accumulated
in an any-hit program.
* [`GeometryFlags::DisableAnyHit`](crate::acceleration::GeometryFlags) - Indicates that traversal should not call
the any-hit program for this primitive even if the corresponding SBT record
contains an any-hit program. Setting this flag usually improves performance
even if no any-hit program is present in the SBT.

Primitives inside a build input are indexed starting from zero. This primitive
index is accessible inside the intersection, any-hit, and closest-hit programs.
If the application chooses to offset this index for all primitives in a build
input, there is no overhead at runtime. This can be particularly useful when
data for consecutive build inputs is stored consecutively in device memory.
The `primitive_index_offset` value is only used when reporting the intersection
primitive.

## Build Flags

An acceleration structure build can be controlled using the values of the
[`BuildFlags`] enum. To enable random vertex access on an acceleration structure,
use [`BuildFlags::ALLOW_RANDOM_VERTEX_ACCESS`](crate::acceleration::BuildFlags). 
To steer trade-offs between build performance, runtime traversal performance
and acceleration structure memory usage, use [`BuildFlags::PREFER_FAST_TRACE`](crate::acceleration::BuildFlags)
and [`BuildFlags::PREFER_FAST_BUILD`](crate::acceleration::BuildFlags). For curve primitives in particular,
these flags control splitting; see “Splitting curve segments”.

The flags [`BuildFlags::PREFER_FAST_TRACE`](crate::acceleration::BuildFlags) and [`BuildFlags::PREFER_FAST_BUILD`](crate::acceleration::BuildFlags)
are mutually exclusive. To combine multiple flags that are not mutually exclusive,
use the logical “or” operator.

## Dynamic Updates

Building an acceleration structure can be computationally costly. Applications
may choose to update an existing acceleration structure using modified vertex
data or bounding boxes. Updating an existing acceleration structure is generally
much faster than rebuilding. However, the quality of the acceleration structure
may degrade if the data changes too much with an update, for example, through
explosions or other chaotic transitions—even if for only parts of the mesh.
The degraded acceleration structure may result in slower traversal performance
as compared to an acceleration structure built from scratch from the modified
input data.

### Dynamic Updates Safe API

The simplest way to use dynamic updates is with the [`DynamicAccel`] structure, which wraps an [`Accel`] and adds extra checks and functionality to support dyanmic updates to the acceleration structure.

Simply call [`DynamicAccel::build()`] as you would with [`Accel`], and then
call [`DynamicAccel::update()`] with the updated build inputs when you want
to update the acceleration structure.

Note that the inputs to [`DynamicAccel::update()`] must have the same structure,
i.e. the number of motion keys, aabbs, triangle topology etc must be the same,
although the underlying data (including the data pointers) can be different.
If the data have a different structure, then behaviour is undefined.
[`DynamicAccel`] checks this by hashing the inputs and returns an error if
the data do not match.

### Dynamic Updates Unsafe API

To allow for future updates of an acceleration structure, set
[`BuildFlags::ALLOW_UPDATE`](crate::acceleration::BuildFlags) in the build flags when building the acceleration
structure initially.

To update the previously built acceleration structure, set the operation to
[`BuildOperation::Update`](crate::acceleration::BuildOperation) and then call [`accel_build()`] on the same output
data. All other options are required to be identical to the original build.
The update is done in-place on the output data.

Updating an acceleration structure usually requires a different amount of temporary memory than the original build.

When updating an existing acceleration structure, only the device pointers and/or
their buffer content may be changed. You cannot change the number of build inputs,
the build input types, build flags, traversable handles for instances (for an
instance-AS), or the number of vertices, indices, AABBs, instances, SBT records
or motion keys. Changes to any of these things may result in undefined behavior,
including GPU faults.

Note the following:

* When using indices, changing the connectivity or, in general, using shuffled
vertex positions will work, but the quality of the acceleration structure will
likely degrade substantially.
* During an animation operation, geometry that should be invisible to the camera
should not be “removed” from the scene, either by moving it very far away or
by converting it into a degenerate form. Such changes to the geometry will also
degrade the acceleration structure.
* In these cases, it is more efficient to re-build the geometry-AS and/or the
instance-AS, or to use the respective masking and flags.

Updating an acceleration structure requires that any other acceleration structure
that is using this acceleration structure as a child directly or indirectly
also needs to be updated or rebuild.

## Compaction
A post-process can compact an acceleration structure after construction. This
process can significantly reduce memory usage, but it requires an additional
pass. The build and compact operations are best performed in batches to ensure
that device synchronization does not degrade performance. The compacted size
depends on the acceleration structure type and its properties and on the device
architecture.

### Compaction Safe API
To compact an [`Accel`] or [`DynamicAccel`] when building, simply pass `true`
for the `compact` parameter. This handles all buffer allocation and management 
internally, providing safely and simplicity at the cost of not being able to re-use
temporary buffers.

### Compaction Unsafe API

To compact the acceleration structure as a post-process, do the following:

* Build flag [`BuildFlags::ALLOW_COMPACTION`](crate::acceleration::BuildFlags) must be set in the
    [`AccelBuildOptions`] passed to optixAccelBuild.
* The emit property [`AccelEmitDesc::CompactedSize`](crate::acceleration::AccelEmitDesc) must be passed to
    [`accel_build()`]. This property is generated on the device and it must be
    copied back to the host if it is required for allocating the new output
    buffer. The application may then choose to compact the acceleration structure
    using [`accel_compact()`].

The [`accel_compact()`] call should be guarded by an
`if compacted_size < output_size` (or similar) to avoid the compacting pass in
cases where it is not beneficial. Note that this check requires a copy of the
compacted size (as queried by [`accel_build()`]) from the device memory to host
memory.

Just like an uncompacted acceleration structure, it is possible to traverse,
update, or relocate a compacted acceleration structure.

For example:
```no_run
use cust::prelude as cu;
use optix::prelude as ox;
# fn doit() -> Result<(), Box<dyn std::error::Error>> {
# cust::init(cu::CudaFlags::empty())?;
# ox::init()?;
# let device = cu::Device::get_device(0)?;
# let cu_ctx = cu::Context::create_and_push(cu::ContextFlags::SCHED_AUTO |
# cu::ContextFlags::MAP_HOST, device)?;
# let ctx = ox::DeviceContext::new(&cu_ctx, false)?;
# let vertices: Vec<[f32; 3]> = Vec::new();
# let indices: Vec<[u32; 3]> = Vec::new();
# let stream = cu::Stream::new(cu::StreamFlags::DEFAULT, None)?;

let buf_vertex = cu::DeviceBuffer::from_slice(&vertices)?;
let buf_indices = cu::DeviceBuffer::from_slice(&indices)?;

let geometry_flags = ox::GeometryFlags::None;

let build_inputs =
    [ox::IndexedTriangleArray::new(
        &[&buf_vertex],
        &buf_indices,
        &[geometry_flags]
    )];

let accel_options =
    ox::AccelBuildOptions::new(
        ox::BuildFlags::ALLOW_COMPACTION,
        ox::BuildOperation::Build
    );

// Get the storage requirements for temporary and output buffers
let sizes = accel_compute_memory_usage(ctx, accel_options, build_inputs)?;

// Allocate temporary and output buffers
let mut output_buffer =
    unsafe { DeviceBuffer::<u8>::uninitialized(sizes.output_size_in_bytes)? };
let mut temp_buffer =
    unsafe { DeviceBuffer::<u8>::uninitialized(sizes.temp_size_in_bytes)? };

// Build the accel
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

stream.synchronize()?;

let mut compacted_size = 0usize;
compacted_size_buffer.copy_to(&mut compacted_size)?;

let accel = if compacted_size < sizes.output_size_in_bytes {
    let mut buf = unsafe { DeviceBuffer::<u8>::uninitialized(compacted_size)? };
    let hnd = unsafe { accel_compact(ctx, stream, hnd, &mut buf)? };

    stream.synchronize()?;
    Accel::from_raw_parts(buf, hnd);
else {
    Accel::from_raw_parts(output_buffer, hnd)
};

# Ok(())
# }
```

## Traversable Objects

### Traversable Objects Safe API

The transform traversable types, [`StaticTransform`](crate::transform::StaticTransform),
[`MatrixMotionTransform`](crate::transform::MatrixMotionTransform), and
[`SrtMotionTransform`](crate::transform::SrtMotionTransform) handle all
necessary memory allocation and pointer conversion for you in their `new()`
constructors.

### Traversable Objects Unsafe API
The instances in an instance-AS may reference transform traversables, as well
as geometry-ASs. Transform traversables are fully managed by the application.
The application needs to create these traversables manually in device memory
in a specific form. The function [`convert_pointer_to_traversable_handle`]
converts a raw pointer into a traversable handle of the specified type. The
traversable handle can then be used to link traversables together.

In device memory, all traversable objects need to be 64-byte aligned. Note that
moving a traversable to another location in memory invalidates the traversable
handle. The application is responsible for constructing a new traversable handle
and updating any other traversables referencing the invalidated traversable
handle.

The traversable handle is considered opaque and the application should not rely
on any particular mapping of a pointer to the traversable handle.

### Traversal of a Single Geometry Acceleration Structure
The traversable handle passed to `optixTrace` can be a traversable handle
created from a geometry-AS. This can be useful for scenes where single
geometry-AS objects represent the root of the scene graph.

If the modules and pipeline only need to support single geometry-AS traversables,
it is beneficial to change the
[`PipelineCompileOptions::traversable_graph_flags`](crate::module::PipelineCompileOptions) from
[`TraversableGraphFlags::ALLOW_ANY`](crate::module::TraversableGraphFlags) to
[`TraversableGraphFlags::ALLOW_SINGLE_GAS`](crate::module::TraversableGraphFlags).

This signals to NVIDIA OptiX 7 that no other traversable types require support
during traversal.

## Motion Blur

Motion support in OptiX targets the rendering of images with motion blur using a
stochastic sampling of time. OptiX supports two types of motion as part of the
scene: transform motion and vertex motion, often called deformation motion. When
setting up the scene traversal graph and building the acceleration structures,
motion options can be specified per acceleration structure as well as per motion
transform traversable. At run time, a time parameter is passed to the trace call
to perform the intersection of a ray against the scene at the selected point in
time.

The general design of the motion feature in OptiX tries to strike a balance
between providing many parameters to offer a high degree of freedom combined
with a simple mapping of scene descriptions to these parameters but also
delivering high traversal performance at the same time. As such OptiX supports
the following key features:

* Vertex and transformation motion
* Matrix as well as SRT (scale rotation translation) transformations
* Arbitrary time ranges (ranges not limited to [0,1]) and flags to specify behavior outside the time range
* Arbitrary concatenations of transformations (for example, a matrix transformation on top of a SRT transformation)
* Per-ray timestamps

Scene descriptions with motion need to map easily to traversable objects and
their motion options as offered by OptiX. As such, the idea is that the motion
options are directly derived by the scene description, delivering high traversal
performance without the need for any performance-driven adjustments. However, due
to the complexity of the subject, there are a few exceptions that are discussed
in this section.

This section details the usage of the motion options on the different traversable
types and how to map scene options best to avoid potential performance pitfalls.

### Basics
Motion is supported by
[`MatrixMotionTransform`],
[`SrtMotionTransform`] and
acceleration structure traversables. The general motion characteristics are
specified per traversable as motion options: the number of motion keys, flags,
and the beginning and ending motion times corresponding to the first and last
key. The remaining motion keys are evenly spaced between the beginning and
ending times. The motion keys are the data at specific points in time and the
data is interpolated in between neighboring keys. The motion options are
specified in the [`MotionOptions`] struct.

The motion options are always specified per traversable (acceleration structure
or motion transform). There is no dependency between the motion options of
traversables; given an instance referencing a geometry acceleration structure
with motion, it is not required to build an instance acceleration structure
with motion. The same goes for motion transforms. Even if an instance references
a motion transform as child traversable, the instance acceleration structure
itself may or may not have motion.

Motion transforms must specify at least two motion keys. Acceleration structures,
however, also accept [`AccelBuildOptions`] with field [`MotionOptions`] set to
`default()`. This effectively disables motion for the acceleration structure and
ignores the motion beginning and ending times, along with the motion flags.

OptiX also supports static transform traversables in addition to the static
transform of an instance. Static transforms are intended for the case of motion
transforms in the scene. Without any motion transforms
([`MatrixMotionTransform`] or
[`SrtMotionTransform`]) in the traversable
graph, any static transformation should be baked into the instance transform.
However, if there is a motion transform, it may be required to apply a static
transformation on a traversable (for example, on a geometry-AS) first before
applying the motion transform. For example, a motion transform may be specified
in world coordinates, but the geometry it applies to needs to be placed into the
scene first (object-to-world transformation, which is usually done using the
instance transform). In this case, a static transform pointing at the geometry
acceleration structure can be used for the object-to-world transformation and
the instance transform pointing to the motion transform has an identity matrix
as transformation.

Motion boundary conditions are specified by using flags. By default, the
behavior for any time outside the time range, is as if time was clamped to the
range, meaning it appears static and visible. Alternatively, to remove the
traversable before the beginning time, set [`MotionFlags::START_VANISH`](crate::acceleration::MotionFlags); to
remove it after the ending time, set [`MotionFlags::END_VANISH`](crate::acceleration::MotionFlags).

For example:
```
let motion_options = MotionFlags {
    num_keys: 3,
    time_begin: -1.0,
    time_end: 1.5
    flags: MotionFlags::NONE,
};
```

OptiX offers two types of motion transforms, SRTs (scale-rotation-translation)
as well as 3x4 affine matrices, each specifying one transform (SRT or matrix)
per motion key. The transformations are always specified as object-to-world
transformation just like the instance transformation. During traversal OptiX
performs a per-component linear interpolation of the two nearest keys. The
rotation component (expressed as a quaternion) of the SRT is an exception,
OptiX ensures that the interpolated quaternion of two SRTs is of unit length
by using nlerp interpolation for performance reasons. This results in a smooth,
scale-preserving rotation in Cartesian space though with non-constant velocity.

For vertex motion, OptiX applies a linear interpolation between the vertex data
that are provided by the application. If intersection programs are used and
AABBs are supplied for the custom primitives, the AABBs are also linearly
interpolated for intersection. The AABBs at the motion keys must therefore be
big enough to contain any motion path of the underlying custom primitive.

There are several device-side functions that take a time parameter such as
`optixTrace` and respect the motion options as set at the traversables. The
result of these device-side functions is always that of the specified point
in time, e.g, the intersection of the ray with the scene at the selected point
in time. Device-side functions are discussed in detail in “Device-side functions”.

### Motion Geometry Acceleration Structure
Use [`Accel::build()`] to build a motion acceleration structure. The motion
options are part of the build options ([`AccelBuildOptions`]) and apply to all
build inputs. Build inputs must specify primitive vertex buffers (for
[`TriangleArray`] and [`CurveArray`]), radius buffers (for [`CurveArray`]), and
AABB buffers (for [`CustomPrimitiveArray`] and [`InstanceArray`]) for all motion
keys. These are interpolated during traversal to obtain the continuous motion vertices and AABBs between the begin and end time.

The motion options are typically defined by the mesh data which should directly
map to the motion options on the geometry acceleration structure. For example,
if a triangle mesh has three per-vertex motion values, the geometry acceleration
structure needs to have three motion keys. Just as for non-motion meshes, it is
possible to combine meshes within a single geometry acceleration structure to
potentially increase traversal performance (this is generally recommended if
there is only a single instance of each mesh and the meshes overlap or are close
together). However, these meshes need to share the same motion options (as they
are specified per geometry acceleration structure). The usual trade-offs apply
in case meshes need to be updated from one frame to another as in an interactive
application. The entire geometry acceleration structure needs to be rebuilt or
refitted if the vertices of at least one mesh change.

It is possible to use a custom intersection program to decouple the actual vertex
data and the motion options of the geometry acceleration structure. Intersection
programs allow any kind of intersection routine. For example, it is possible to
implement a three-motion-key-triangle intersection, but build a static geometry
acceleration structure over AABBs by passing AABBs to the geometry acceleration
structure build that enclose the full motion path of the triangles. However, this
is generally not recommended for two reasons: First, the AABBs tend to increase
in size very quickly even with very little motion. Second, it prevents the use
of hardware intersection routines. Both of these effects can have a tremendous
impact on performance.

### Motion Instance Acceleration Structure

Just as for a geometry acceleration structure, the motion options for an instance acceleration structure are specified as part of the build options. The notable difference to a geometry acceleration structure is that the motion options for an instance acceleration structure almost only impact performance. Hence, whether or not to build a motion instance acceleration structure has no impact on the correctness of the rendering (determining which instances can be intersected), but impacts memory usage as well as traversal performance. The only exception to that are the vanish flags as these force any instance of the instance acceleration structure to be non-intersectable for any ray time outside of the time range of the instance acceleration structure.

In the following, guidelines are provided on setting the motion options to achieve good performance and avoid pitfalls. We will focus on the number of motion keys, usually the main discriminator for traversal performance and the only factor for memory usage. The optimal number of motion keys used for the instance acceleration structure build depends on the amount and linearity of the motion of the traversables referenced by the instances. The time beginning and ending range are usually defined by what is required to render the current frame. The recommendations given here may change in the future.

The following advice should be considered a simplified heuristic. A more detailed derivation of whether or not to use motion is given below. For RTCores version 1.0 (Turing architecture), do not use motion for instance acceleration structure, but instead build a static instance acceleration structure that can leverage hardware-accelerated traversal. For any other device (devices without RTCores or RTCores version >= 2.0), build a motion instance acceleration structure if any of the instances references a motion transform or a motion acceleration structure as traversable child.

If a motion instance acceleration structure is built, it is often sufficient to use a low number of motion keys (two or three) to avoid high memory costs. Also, it is not required to use a large number of motion keys just because one of the referenced motion transforms has many motion keys (such as the maximum motion keys of any referenced traversable by any of the instances). The motion options have no dependency between traversable objects and a high number of motion keys on the instance acceleration structure causes a high memory overhead. Clearly, motion should not be used for an instance acceleration structure if the instances only reference static traversables.

Further considerations when using motion blur:

#### Is motion enabled?
An instance acceleration structure should be built with motion on (the number of motion keys larger than one) if the overall amount of motion of the instanced traversables is non-minimal. For a single instance this can be quantified by the amount of change of its AABB over time. Hence, in case of a simple translation (for example, due to a matrix motion transform), the metric is the amount of the translation in comparison to the size of the AABB. In case of a scaling, it is the ratio of the size of the AABB at different points in times. If sufficiently many instanced traversables exhibit a non-minimal amount of change of their AABB over time, build a motion instance acceleration structure. Inversely, a static instance acceleration structure can yield higher traversal performance if many instanced traversables have no motion at all or only very little. The latter can happen for rotations. A rotation around the center of an object causes a rather small difference in the AABB of the object. However, if the rotational pivot point is not the center, it is likely to cause a big difference in the AABB of the object.

As it is typically hard to actually quantify the amount of motion for the instances, switch to motion if sufficiently many instanced traversables have or are expected to have motion. Yet it is difficult to predict when exactly it pays off to use or not use motion on the instance acceleration structure.

#### If motion is enabled, how many keys should be defined?

A reasonable metric to determine the required number of motion keys for an instance acceleration structure is the linearity of the motion of the instanced traversables. If there are motion transforms with many motion keys, rotations, or a hierarchical set of motion transforms, more motion keys on the instance acceleration structure may increase traversal performance. Transformations like a simple translation, rotation around the center of an object, a small scale, or even all of those together are usually handles well by a two-motion-key instance acceleration structure.

Finally, the quality of the instance acceleration structure is also affected by the number of motion keys of the referenced traversables of the instances. As such, it is desirable to have the motion options of the instance acceleration structure match the motion options of any referenced motion transform. For example, if all instances reference motion transforms with three keys, it is reasonable to also use three motion keys for the instance acceleration structure. Note that also in this case the statement from above still applies that using more motion keys only helps if the underlying transformation results in a non-linear motion.

### Motion Matrix Transform

The motion matrix transform traversable ([`MatrixMotionTransform`]) transforms the ray during traversal using a motion matrix. The traversable provides a 3x4 row-major object-to-world transformation matrix for each motion key. The final motion matrix is constructed during traversal by interpolating the elements of the matrices at the nearest motion keys.

The [`MatrixMotionTransform`] can be created with an arbitrary number of keys
using its [`new()`](crate::acceleration::MatrixMotionTransform::new) constructor.

### Motion Scale Rotate Translate Transform

The behavior of the motion transform [`SrtMotionTransform`] is similar to the matrix motion transform [`MatrixMotionTransform`]. In [`SrtMotionTransform`] the object-to-world transforms per motion key are specified as a scale, rotation and translation (SRT) decomposition instead of a single 3x4 matrix. Each motion key is a struct of type [`SrtData`], which consists of 16 floats:

```
struct SrtData {
   pub sx: f32,
   pub a: f32,
   pub b: f32,
   pub pvx: f32,
   pub sy: f32,
   pub c: f32,
   pub pvy: f32,
   pub sz: f32,
   pub pvz: f32,
   pub qx: f32,
   pub qy: f32,
   pub qz: f32,
   pub qw: f32,
   pub tx: f32,
   pub ty: f32,
   pub tz: f32,
}
```

* The scaling matrix,
$$
S=\begin{bmatrix}
sx & a & b & pvx \cr 0 & sy & c & pvy \cr 0 & 0 & sz & pvz
\end{bmatrix}
$$

defines an affine transformation that can include scale, shear, and a translation.
The translation allows to define the pivot point for the subsequent rotation.

* The rotation quaternion
$$
R = [qx, qy, qz, qw]
$$
describes a rotation with angular
component $qw = \cos(\theta / 2)$ and other components
$$
[qx, qy, qz] = \sin(\theta / 2) \cdot [ax, ay, az]
$$ where the axis $[ax, ay, az]$ is normalized.

* The translation matrix,
$$
T = \begin{bmatrix} 1 & 0 & 0 & tx \cr 0 & 1 & 0 & ty \cr 0 & 0 & 1 & tz \end{bmatrix}
$$
defines another translation that is applied after the rotation. Typically, this
translation includes the inverse translation from the matrix $S$ to reverse the
translation for the pivot point for $R$.

To obtain the effective transformation at time $t$, the elements of the components
of $S$, $R$, and $T$ will be interpolated linearly. The components are then
multiplied to obtain the combined transformation $C = T \times R \times S$. The
transformation $C$ is the effective object-to-world transformations at time $t$,
and $C^{-1}$ is the effective world-to-object transformation at time $t$.

#### Example 1 - rotation about the origin:

Use two motion keys. Set the first key to identity values. For the second key, define a quaternion from an axis and angle, for example, a 60-degree rotation about the z axis is given by:

$$ Q = [ 0 0 \sin(\pi/6) \cos(\pi/6) ] $$

#### Example 2 - rotation about a pivot point:
Use two motion keys. Set the first key to identity values. Represent the pivot point as a translation $P$, and define the second key as follows:
$$
S^{\prime} = P^{-1} \times S \newline
T^{\prime} = T \times P \newline
C = T^{\prime} \times R \times S^{\prime}
$$

#### Example 3 - scaling about a pivot point

Use two motion keys. Set the first key to identity values. Represent the pivot as a translation $G = [G_x, G_y, f G_z]$ and modify the pivot point described above:

$$
P_x^{\prime} = P_x + (-S_x \times G_x + G_x) \newline
P_y^{\prime} = P_y + (-S_y \times G_y + G_y) \newline
P_z^{\prime} = P_z + (-S_z \times G_z + G_z) \newline
$$

### Transforms trade-offs
Several trade-offs must be considered when using transforms.

#### SRTs compared to matrix motion transforms
Use SRTs for any transformations containing a rotation. Only SRTs produce a smooth rotation without distortion. They also avoid any oversampling of matrix transforms to approximate a rotation. However, note that the maximum angle of rotation due to two neighboring SRT keys needs to be less than 180 degrees, hence, the dot product of the quaternions needs to be positive. This way the rotations are interpolated using the shortest path. If a rotation of 180 degrees or more is required, additional keys need to be specified such that the rotation between two keys is less than 180 degrees. OptiX uses nlerp to interpolate quaternion at runtime. While nlerp produces the best traversal performance, it causes non-constant velocity in the rotation. The variation of rotational velocity is directly dependent on the amount of the rotation. If near constant rotation velocity is required, more SRT keys can be used.

Due to the complexity of the rotation, instance acceleration structure builds with instances that reference SRT transforms can be relatively slow. For real-time or interactive applications, it can be advantageous to use matrix transforms to have fast rebuilds or refits of the instance acceleration structure.

#### Motion options for motion transforms
The motion options for motion transforms should be derived by the scene setup and used as needed. The number of keys is defined by the number of transformations specified by the scene description. The beginning, ending times should be as needed for the frame or tighter if specified by the scene description.

Avoid duplicating instances of motion transforms to achieve a motion behavior that can also be expressed by a single motion transform but many motion keys. An example is the handling of irregular keys, which is discussed in the following section.

#### Dealing with irregular keys
OptiX only supports regular time intervals in its motion options. Irregular keys should be resampled to fit regular keys, potentially with a much higher number of keys if needed.

A practical example for this is a motion matrix transform that performs a rotation. Since the matrix elements are linearly interpolated between keys, the rotation is not an actual rotation, but a scale/shear/translation. To avoid visual artifacts, the rotation needs to be sampled with potentially many matrix motion keys. Such a sampling bounds the maximum error in the approximation of the rotation by the linear interpolation of matrices. The sampling should not try to minimize the number of motion keys by outputting irregular motion keys, but rather oversample the rotation with many keys.

Duplicate motion transforms should not be used as a workaround for irregular keys, where each key has varying motion beginning and ending times and vanish motion flags set. This duplication creates traversal overhead as all copies need to be intersected and their motion times compared to the ray's time.


</details>

[`Accel::build()`]: crate::acceleration::Accel::build
[`Accel::from_raw_parts()`]: crate::acceleration::Accel::from_raw_parts
[`Accel]: crate::acceleration::Accel
[`Instance`]: crate::instance_array::Instance
[`TriangleArray`]: crate::triangle_array::TriangleArray
[`CurveArray`]: crate::curve_array::CurveArray
[`InstanceArray`]: crate::instance_array::InstanceArray
[`MatrixMotionTransform`]: crate::transform::MatrixMotionTransform
[`SrtMotionTransform`]: crate::transform::SrtMotionTransform
[`BuildInput`]: crate::acceleration::BuildInput
[`TraversableHandle`]: crate::acceleration::TraversableHandle
[`accel_build()`]: crate::acceleration::accel_build
[`accel_compute_memory_usage()`]: crate::acceleration::accel_compute_memory_usage
[`accel_compact()`]: crate::acceleration::accel_compact
[`GeometryFlags`]: crate::acceleration::GeometryFlags
[`BuildFlags`]: crate::acceleration::BuildFlags
[`DynamicAccel`]: crate::acceleration::DynamicAccel
[`DynamicAccel::build()`]: crate::acceleration::DynamicAccel::build
[`DynamicAccel::update()`]: crate::acceleration::DynamicAccel::update
[`AccelBuildOptions`]: crate::acceleration::AccelBuildOptions
[`convert_pointer_to_traversable_handle`]: crate::acceleration::convert_pointer_to_traversable_handle
[`MotionOptions`]: crate::acceleration::MotionOptions
