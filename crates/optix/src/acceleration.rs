use crate::{
    context::DeviceContext, error::Error, instance_array::BuildInputInstanceArray, module::Module,
    optix_call, sys, triangle_array::BuildInputTriangleArray,
};
use cust::{
    memory::{DBox, DBuffer, DSlice},
    DeviceCopy,
};
type Result<T, E = Error> = std::result::Result<T, E>;

/// Opaque handle to a traversable acceleration structure.
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

impl DeviceContext {
    /// Computes the device memory required for temporary and output buffers
    /// when building the acceleration structure. Use the returned sizes to
    /// allocate enough memory to pass to `accel_build()`
    pub fn accel_compute_memory_usage<T: BuildInputTriangleArray, I: BuildInputInstanceArray>(
        &self,
        accel_options: &[AccelBuildOptions],
        build_inputs: &[BuildInput<T, I>],
    ) -> Result<AccelBufferSizes> {
        let mut buffer_sizes = AccelBufferSizes::default();
        let build_sys: Vec<_> = build_inputs
            .iter()
            .map(|b| match b {
                BuildInput::TriangleArray(bita) => sys::OptixBuildInput {
                    type_: sys::OptixBuildInputType_OPTIX_BUILD_INPUT_TYPE_TRIANGLES,
                    input: sys::OptixBuildInputUnion {
                        triangle_array: std::mem::ManuallyDrop::new(bita.to_sys()),
                    },
                },
                BuildInput::InstanceArray(biia) => sys::OptixBuildInput {
                    type_: sys::OptixBuildInputType_OPTIX_BUILD_INPUT_TYPE_INSTANCES,
                    input: sys::OptixBuildInputUnion {
                        instance_array: std::mem::ManuallyDrop::new(biia.to_sys()),
                    },
                },
                _ => unimplemented!(),
            })
            .collect();

        unsafe {
            Ok(optix_call!(optixAccelComputeMemoryUsage(
                self.raw,
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
    pub fn accel_build<T: BuildInputTriangleArray, I: BuildInputInstanceArray>(
        &self,
        stream: &cust::stream::Stream,
        accel_options: &[AccelBuildOptions],
        build_inputs: &[BuildInput<T, I>],
        temp_buffer: &mut DSlice<u8>,
        output_buffer: &mut DSlice<u8>,
        emitted_properties: &mut [AccelEmitDesc],
    ) -> Result<TraversableHandle> {
        let mut traversable_handle = TraversableHandle { inner: 0 };
        let properties: Vec<sys::OptixAccelEmitDesc> =
            emitted_properties.iter_mut().map(|p| p.into()).collect();

        let build_sys: Vec<_> = build_inputs
            .iter()
            .map(|b| match b {
                BuildInput::TriangleArray(bita) => sys::OptixBuildInput {
                    type_: sys::OptixBuildInputType_OPTIX_BUILD_INPUT_TYPE_TRIANGLES,
                    input: sys::OptixBuildInputUnion {
                        triangle_array: std::mem::ManuallyDrop::new(bita.to_sys()),
                    },
                },
                BuildInput::InstanceArray(biia) => sys::OptixBuildInput {
                    type_: sys::OptixBuildInputType_OPTIX_BUILD_INPUT_TYPE_INSTANCES,
                    input: sys::OptixBuildInputUnion {
                        instance_array: std::mem::ManuallyDrop::new(biia.to_sys()),
                    },
                },
                _ => unimplemented!(),
            })
            .collect();

        unsafe {
            Ok(optix_call!(optixAccelBuild(
                self.raw,
                stream.as_inner(),
                accel_options.as_ptr() as *const _,
                build_sys.as_ptr(),
                build_sys.len() as u32,
                temp_buffer.as_device_ptr().as_raw_mut() as u64,
                temp_buffer.len(),
                output_buffer.as_device_ptr().as_raw_mut() as u64,
                output_buffer.len(),
                &mut traversable_handle as *mut _ as *mut _,
                properties.as_ptr() as *const _,
                properties.len() as u32,
            ))
            .map(|_| traversable_handle)?)
        }
    }

    /// Compacts the acceleration structure referenced by `input_handle`,
    /// storing the result in `output_buffer` and returning a handle to the
    /// newly compacted structure
    pub fn accel_compact(
        &self,
        stream: &cust::stream::Stream,
        input_handle: TraversableHandle,
        output_buffer: &mut DSlice<u8>,
    ) -> Result<TraversableHandle> {
        let mut traversable_handle = TraversableHandle { inner: 0 };
        unsafe {
            Ok(optix_call!(optixAccelCompact(
                self.raw,
                stream.as_inner(),
                input_handle.inner,
                output_buffer.as_device_ptr().as_raw_mut() as u64,
                output_buffer.len(),
                &mut traversable_handle as *mut _ as *mut _,
            ))
            .map(|_| traversable_handle)?)
        }
    }
}

bitflags::bitflags! {
    pub struct BuildFlags: u32 {
        const NONE = sys::OptixBuildFlags_OPTIX_BUILD_FLAG_NONE;
        const ALLOW_UPDATE = sys::OptixBuildFlags_OPTIX_BUILD_FLAG_ALLOW_UPDATE;
        const ALLOW_COMPACTION = sys::OptixBuildFlags_OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
        const PREFER_FAST_TRACE = sys::OptixBuildFlags_OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
        const PREFER_FAST_BUILD = sys::OptixBuildFlags_OPTIX_BUILD_FLAG_PREFER_FAST_BUILD;
        const ALLOW_RANDOM_VERTEX_ACCESS = sys::OptixBuildFlags_OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS;
    }
}

#[repr(u32)]
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum BuildOperation {
    Build = sys::OptixBuildOperation_OPTIX_BUILD_OPERATION_BUILD,
    Update = sys::OptixBuildOperation_OPTIX_BUILD_OPERATION_UPDATE,
}

bitflags::bitflags! {
    pub struct MotionFlags: u16 {
        const NONE = sys::OptixMotionFlags_OPTIX_MOTION_FLAG_NONE as u16;
        const START_VANISH = sys::OptixMotionFlags_OPTIX_MOTION_FLAG_START_VANISH as u16;
        const END_VANISH = sys::OptixMotionFlags_OPTIX_MOTION_FLAG_END_VANISH as u16;
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct MotionOptions {
    pub num_keys: u16,
    pub flags: MotionFlags,
    pub time_begin: f32,
    pub time_end: f32,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct AccelBuildOptions {
    build_flags: BuildFlags,
    operation: BuildOperation,
    motion_options: MotionOptions,
}

impl AccelBuildOptions {
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

    pub fn num_keys(mut self, num_keys: u16) -> Self {
        self.motion_options.num_keys = num_keys;
        self
    }

    pub fn time_interval(mut self, time_begin: f32, time_end: f32) -> Self {
        self.motion_options.time_begin = time_begin;
        self.motion_options.time_end = time_end;
        self
    }

    pub fn motion_flags(mut self, flags: MotionFlags) -> Self {
        self.motion_options.flags = flags;
        self
    }
}

#[repr(C)]
#[derive(Debug, Default, Copy, Clone, PartialEq)]
pub struct AccelBufferSizes {
    pub output_size_in_bytes: usize,
    pub temp_size_in_bytes: usize,
    pub temp_update_size_in_bytes: usize,
}

pub struct TriangleArrayDefault;
impl BuildInputTriangleArray for TriangleArrayDefault {
    fn to_sys(&self) -> sys::OptixBuildInputTriangleArray {
        unreachable!()
    }
}
pub struct InstanceArrayDefault;
impl BuildInputInstanceArray for InstanceArrayDefault {
    fn to_sys(&self) -> sys::OptixBuildInputInstanceArray {
        unreachable!()
    }
}

pub enum BuildInput<
    T: BuildInputTriangleArray = TriangleArrayDefault,
    I: BuildInputInstanceArray = InstanceArrayDefault,
> {
    TriangleArray(T),
    CurveArray,
    CustomPrimitiveArray,
    InstanceArray(I),
}

pub enum AccelEmitDesc {
    CompactedSize(DBox<u64>),
    Aabbs(DBuffer<Aabb>), //< FIXME: need to handle OptixAabbBufferByteAlignment here
}

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
                result: p.as_device_ptr().as_raw() as u64,
                type_: sys::OptixAccelPropertyType_OPTIX_PROPERTY_TYPE_COMPACTED_SIZE,
            },
            AccelEmitDesc::Aabbs(p) => Self {
                result: p.as_device_ptr().as_raw() as u64,
                type_: sys::OptixAccelPropertyType_OPTIX_PROPERTY_TYPE_AABBS,
            },
        }
    }
}

#[repr(u32)]
#[derive(Copy, Clone, PartialEq)]
pub enum GeometryFlags {
    None = sys::OptixGeometryFlags::None as u32,
    DisableAnyHit = sys::OptixGeometryFlags::DisableAnyHit as u32,
    RequireSingleAnyHitCall = sys::OptixGeometryFlags::RequireSingleAnyHitCall as u32,
}
