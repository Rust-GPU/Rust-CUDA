use crate::{
    acceleration::{BuildInput, GeometryFlags},
    error::Error,
    sys,
};
use cust::memory::DeviceSlice;
use cust_raw::CUdeviceptr;
type Result<T, E = Error> = std::result::Result<T, E>;

use std::marker::PhantomData;

pub struct CurveArray<'v, 'w, 'i> {
    curve_type: CurveType,
    num_primitives: u32,
    vertex_buffers: PhantomData<&'v f32>,
    num_vertices: u32,
    d_vertex_buffers: Vec<CUdeviceptr>,
    vertex_stride_in_bytes: u32,
    width_buffers: PhantomData<&'w f32>,
    num_width_buffers: u32,
    d_width_buffers: Vec<CUdeviceptr>,
    width_stride_in_bytes: u32,
    index_buffer: &'i DeviceSlice<u32>,
    index_stride_in_bytes: u32,
    flags: GeometryFlags,
    primitive_index_offset: u32,
}

impl<'v, 'w, 'i> CurveArray<'v, 'w, 'i> {
    pub fn new(
        curve_type: CurveType,
        num_primitives: u32,
        vertex_buffers: &[&'v DeviceSlice<f32>],
        width_buffers: &[&'w DeviceSlice<f32>],
        index_buffer: &'i DeviceSlice<u32>,
    ) -> Result<CurveArray<'v, 'w, 'i>> {
        // TODO (AL): Do some sanity checking on the values here
        let num_vertices = vertex_buffers[0].len() as u32;
        let d_vertex_buffers: Vec<_> = vertex_buffers.iter().map(|b| b.as_device_ptr()).collect();

        let num_width_buffers = width_buffers.len() as u32;
        let d_width_buffers: Vec<_> = width_buffers.iter().map(|b| b.as_device_ptr()).collect();

        Ok(CurveArray {
            curve_type,
            num_primitives,
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

    pub fn vertex_stride(mut self, stride_in_bytes: u32) -> Self {
        self.vertex_stride_in_bytes = stride_in_bytes;
        self
    }

    pub fn width_stride(mut self, stride_in_bytes: u32) -> Self {
        self.vertex_stride_in_bytes = stride_in_bytes;
        self
    }

    pub fn index_stride(mut self, stride_in_bytes: u32) -> Self {
        self.vertex_stride_in_bytes = stride_in_bytes;
        self
    }

    pub fn flags(mut self, flags: GeometryFlags) -> Self {
        self.flags = flags;
        self
    }

    pub fn primitive_index_offset(mut self, offset: u32) -> Self {
        self.primitive_index_offset = offset;
        self
    }
}

impl<'v, 'w, 'i> BuildInput for CurveArray<'v, 'w, 'i> {
    fn to_sys(&self) -> sys::OptixBuildInput {
        sys::OptixBuildInput {
            type_: sys::OptixBuildInputType_OPTIX_BUILD_INPUT_TYPE_CURVES,
            input: sys::OptixBuildInputUnion {
                curve_array: std::mem::ManuallyDrop::new(sys::OptixBuildInputCurveArray {
                    curveType: self.curve_type.into(),
                    numPrimitives: self.num_primitives,
                    vertexBuffers: self.d_vertex_buffers.as_ptr() as *const CUdeviceptr,
                    numVertices: self.num_vertices,
                    vertexStrideInBytes: self.vertex_stride_in_bytes,
                    widthBuffers: self.d_width_buffers.as_ptr() as *const CUdeviceptr,
                    widthStrideInBytes: self.width_stride_in_bytes,
                    normalBuffers: std::ptr::null(),
                    normalStrideInBytes: 0,
                    indexBuffer: self.index_buffer.as_device_ptr(),
                    indexStrideInBytes: self.index_stride_in_bytes,
                    flag: self.flags as u32,
                    primitiveIndexOffset: self.primitive_index_offset,
                }),
            },
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub enum CurveType {
    RoundLinear,
    RoundQuadraticBSpline,
    RoundCubicBSpline,
}

impl From<CurveType> for sys::OptixPrimitiveType {
    fn from(c: CurveType) -> Self {
        match c {
            CurveType::RoundLinear => sys::OptixPrimitiveType_OPTIX_PRIMITIVE_TYPE_ROUND_LINEAR,
            CurveType::RoundQuadraticBSpline => {
                sys::OptixPrimitiveType_OPTIX_PRIMITIVE_TYPE_ROUND_QUADRATIC_BSPLINE
            }
            CurveType::RoundCubicBSpline => {
                sys::OptixPrimitiveType_OPTIX_PRIMITIVE_TYPE_ROUND_CUBIC_BSPLINE
            }
        }
    }
}
