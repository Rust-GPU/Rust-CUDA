use std::marker::PhantomData;

use crate::{
    acceleration::{BuildInput, GeometryFlags},
    sys,
};
use cust::memory::DeviceSlice;
use cust_raw::CUdeviceptr;

#[repr(u32)]
#[derive(Copy, Clone, PartialEq)]
pub enum VertexFormat {
    None = sys::OptixVertexFormat_OPTIX_VERTEX_FORMAT_NONE as u32,
    Float3 = sys::OptixVertexFormat_OPTIX_VERTEX_FORMAT_FLOAT3 as u32,
    Float2 = sys::OptixVertexFormat_OPTIX_VERTEX_FORMAT_FLOAT2 as u32,
    Half3 = sys::OptixVertexFormat_OPTIX_VERTEX_FORMAT_HALF3 as u32,
    Half2 = sys::OptixVertexFormat_OPTIX_VERTEX_FORMAT_HALF2 as u32,
    SNorm16 = sys::OptixVertexFormat_OPTIX_VERTEX_FORMAT_SNORM16_3 as u32,
    SNorm32 = sys::OptixVertexFormat_OPTIX_VERTEX_FORMAT_SNORM16_2 as u32,
}

#[repr(u32)]
#[derive(Copy, Clone, PartialEq)]
pub enum IndicesFormat {
    None = sys::OptixIndicesFormat_OPTIX_INDICES_FORMAT_NONE as u32,
    Short3 = sys::OptixIndicesFormat_OPTIX_INDICES_FORMAT_UNSIGNED_SHORT3 as u32,
    Int3 = sys::OptixIndicesFormat_OPTIX_INDICES_FORMAT_UNSIGNED_INT3 as u32,
}

#[repr(u32)]
#[derive(Copy, Clone, PartialEq)]
pub enum TransformFormat {
    None = sys::OptixTransformFormat_OPTIX_TRANSFORM_FORMAT_NONE,
    MatrixFloat12 = sys::OptixTransformFormat_OPTIX_TRANSFORM_FORMAT_MATRIX_FLOAT12,
}

pub trait Vertex: cust::memory::DeviceCopy {
    const FORMAT: VertexFormat;
    const STRIDE: u32 = 0;
}

#[cfg(feature = "half")]
impl Vertex for [half::f16; 2] {
    const FORMAT: VertexFormat = VertexFormat::Half2;
}

#[cfg(feature = "half")]
impl Vertex for [half::f16; 3] {
    const FORMAT: VertexFormat = VertexFormat::Half3;
}

#[cfg(feature = "half")]
impl Vertex for mint::Vector2<f16> {
    const FORMAT: VertexFormat = VertexFormat::Half2;
}

#[cfg(feature = "half")]
impl Vertex for mint::Vector3<f16> {
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

pub struct TriangleArray<'v, 'g, V: Vertex> {
    // We hold slices here to make sure the referenced device memory remains
    // valid for the lifetime of the build input
    vertex_buffers: PhantomData<&'v V>,
    num_vertices: u32,
    d_vertex_buffers: Vec<CUdeviceptr>,
    // per-sbt-record geometry flags
    geometry_flags: &'g [GeometryFlags],
}

impl<'v, 'g, V: Vertex> TriangleArray<'v, 'g, V> {
    pub fn new(vertex_buffers: &[&'v DeviceSlice<V>], geometry_flags: &'g [GeometryFlags]) -> Self {
        // TODO (AL): do some sanity checking on the slice lengths here
        let num_vertices = vertex_buffers[0].len() as u32;
        let d_vertex_buffers: Vec<_> = vertex_buffers.iter().map(|b| b.as_device_ptr()).collect();
        TriangleArray {
            vertex_buffers: PhantomData,
            num_vertices,
            d_vertex_buffers,
            geometry_flags,
        }
    }
}

impl<'v, 'g, V: Vertex> BuildInput for TriangleArray<'v, 'g, V> {
    fn to_sys(&self) -> sys::OptixBuildInput {
        sys::OptixBuildInput {
            type_: sys::OptixBuildInputType_OPTIX_BUILD_INPUT_TYPE_TRIANGLES,
            input: sys::OptixBuildInputUnion {
                triangle_array: std::mem::ManuallyDrop::new(sys::OptixBuildInputTriangleArray {
                    vertexBuffers: self.d_vertex_buffers.as_ptr() as *const u64,
                    numVertices: self.num_vertices,
                    vertexFormat: V::FORMAT as u32,
                    vertexStrideInBytes: V::STRIDE,
                    indexBuffer: 0,
                    numIndexTriplets: 0,
                    indexFormat: 0,
                    indexStrideInBytes: 0,
                    preTransform: 0,
                    flags: self.geometry_flags.as_ptr() as *const _,
                    numSbtRecords: 1,
                    sbtIndexOffsetBuffer: 0,
                    sbtIndexOffsetSizeInBytes: 0,
                    sbtIndexOffsetStrideInBytes: 0,
                    primitiveIndexOffset: 0,
                    transformFormat: sys::OptixTransformFormat_OPTIX_TRANSFORM_FORMAT_NONE,
                }),
            },
        }
    }
}

pub struct IndexedTriangleArray<'v, 'g, 'i, V: Vertex, I: IndexTriple> {
    // We hold slices here to make sure the referenced device memory remains
    // valid for the lifetime of the build input
    vertex_buffers: PhantomData<&'v V>,
    num_vertices: u32,
    d_vertex_buffers: Vec<CUdeviceptr>,
    index_buffer: &'i DeviceSlice<I>,
    // per-object geometry flags
    geometry_flags: &'g [GeometryFlags],
}

impl<'v, 'g, 'i, V: Vertex, I: IndexTriple> IndexedTriangleArray<'v, 'g, 'i, V, I> {
    pub fn new(
        vertex_buffers: &[&'v DeviceSlice<V>],
        index_buffer: &'i DeviceSlice<I>,
        geometry_flags: &'g [GeometryFlags],
    ) -> Self {
        let num_vertices = vertex_buffers[0].len() as u32;
        let d_vertex_buffers: Vec<_> = vertex_buffers.iter().map(|b| b.as_device_ptr()).collect();
        IndexedTriangleArray {
            vertex_buffers: PhantomData,
            num_vertices,
            d_vertex_buffers,
            geometry_flags,
            index_buffer,
        }
    }
}

impl<'v, 'g, 'i, V: Vertex, I: IndexTriple> BuildInput for IndexedTriangleArray<'v, 'g, 'i, V, I> {
    fn to_sys(&self) -> sys::OptixBuildInput {
        sys::OptixBuildInput {
            type_: sys::OptixBuildInputType_OPTIX_BUILD_INPUT_TYPE_TRIANGLES,
            input: sys::OptixBuildInputUnion {
                triangle_array: std::mem::ManuallyDrop::new(sys::OptixBuildInputTriangleArray {
                    vertexBuffers: self.d_vertex_buffers.as_ptr() as *const u64,
                    numVertices: self.num_vertices,
                    vertexFormat: V::FORMAT as u32,
                    vertexStrideInBytes: V::STRIDE,
                    indexBuffer: self.index_buffer.as_device_ptr(),
                    numIndexTriplets: self.index_buffer.len() as u32,
                    indexFormat: I::FORMAT as u32,
                    indexStrideInBytes: I::STRIDE,
                    preTransform: 0,
                    flags: self.geometry_flags.as_ptr() as *const _,
                    numSbtRecords: 1,
                    sbtIndexOffsetBuffer: 0,
                    sbtIndexOffsetSizeInBytes: 0,
                    sbtIndexOffsetStrideInBytes: 0,
                    primitiveIndexOffset: 0,
                    transformFormat: sys::OptixTransformFormat_OPTIX_TRANSFORM_FORMAT_NONE,
                }),
            },
        }
    }
}
