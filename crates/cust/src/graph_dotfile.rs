/// Implementation of turning a Graph into a dotfile for debugging and visualization.
use crate::{
    error::CudaResult,
    graph::{Graph, GraphNode, GraphNodeType},
};
use std::fmt::Write;

// CUDA has a function exactly for this, but it has a couple issues:
// - it includes useless info users dont really need
// - it can only dump it to a file
// - it takes a cstring for the

// #[allow(unused_must_use)]
// pub(crate) fn graph_to_dot(graph: &mut Graph) -> CudaResult<String> {
//     let mut dot = String::new();

//     writeln!(dot, "digraph dot {{");
//     writeln!(dot, "  subgraph cluster_1 {{");
//     writeln!(dot, "    label = \"Graph 1\" graph[style = \"dashed\"];");
//     writeln!(dot, "    ")

//     Ok(dot)
// }

// fn node_to_dot(graph: &mut Graph, node: GraphNode, graph_num: u32, node_num: u32) -> CudaResult<String> {
//     let kind = graph.node_type(node)?;

//     let name = format!("graph_{}_node_{}", graph_num, node_num);
//     let style = "Mrecord";
//     let contents = match graph.node_type(node)? {
//         GraphNodeType::KernelInvocation => {
//             let params = graph.kernel_node_params(node)?;

//         }
//     }
// }
