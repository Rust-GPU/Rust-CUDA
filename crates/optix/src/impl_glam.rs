use crate::acceleration::{IndexTriple, IndicesFormat, Vertex, VertexFormat};

impl Vertex for glam::Vec3 {
    const FORMAT: VertexFormat = VertexFormat::Float3;
}

impl IndexTriple for glam::IVec3 {
    const FORMAT: IndicesFormat = IndicesFormat::Int3;
}
