use crate::{
    acceleration::{Aabb, BuildInput, GeometryFlags},
    error::Error,
    sys,
};
use cust::memory::{DeviceSlice, GpuBox};
use cust_raw::CUdeviceptr;
type Result<T, E = Error> = std::result::Result<T, E>;

use std::hash::Hash;
use std::marker::PhantomData;

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

impl<'a, 'g, 's> Hash for CustomPrimitiveArray<'a, 's> {
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
        let aabb_buffers: Vec<_> = aabb_buffers.iter().map(|b| b.as_device_ptr()).collect();

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

impl<'a, 's> BuildInput for CustomPrimitiveArray<'a, 's> {
    fn to_sys(&self) -> sys::OptixBuildInput {
        sys::OptixBuildInput {
            type_: sys::OptixBuildInputType_OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES,
            input: sys::OptixBuildInputUnion {
                custom_primitive_array: std::mem::ManuallyDrop::new(
                    sys::OptixBuildInputCustomPrimitiveArray {
                        aabbBuffers: self.aabb_buffers.as_ptr(),
                        numPrimitives: self.num_primitives,
                        strideInBytes: self.stride_in_bytes,
                        flags: self.flags.as_ptr() as *const u32,
                        numSbtRecords: self.num_sbt_records,
                        sbtIndexOffsetBuffer: if let Some(sbt_index_offset_buffer) =
                            self.sbt_index_offset_buffer
                        {
                            sbt_index_offset_buffer.as_device_ptr()
                        } else {
                            0
                        },
                        sbtIndexOffsetSizeInBytes: 4,
                        sbtIndexOffsetStrideInBytes: self.sbt_index_offset_stride_in_bytes,
                        primitiveIndexOffset: self.primitive_index_offset,
                    },
                ),
            },
        }
    }
}
