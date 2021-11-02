use crate::{
    context::DeviceContext,
    error::Error,
    module::{CompileDebugLevel, PipelineCompileOptions},
    optix_call,
    program_group::ProgramGroup,
    sys,
};
type Result<T, E = Error> = std::result::Result<T, E>;

use std::ffi::CStr;

#[repr(transparent)]
pub struct Pipeline {
    pub(crate) raw: sys::OptixPipeline,
}

#[repr(C)]
#[derive(Debug, Hash, PartialEq, Copy, Clone)]
pub struct PipelineLinkOptions {
    pub max_trace_depth: u32,
    pub debug_level: CompileDebugLevel,
}

impl From<PipelineLinkOptions> for sys::OptixPipelineLinkOptions {
    fn from(o: PipelineLinkOptions) -> Self {
        sys::OptixPipelineLinkOptions {
            maxTraceDepth: o.max_trace_depth,
            debugLevel: o.debug_level as u32,
        }
    }
}

/// # Creating and destroying `Pipeline`s
impl Pipeline {
    pub fn new(
        ctx: &mut DeviceContext,
        pipeline_compile_options: &PipelineCompileOptions,
        link_options: PipelineLinkOptions,
        program_groups: &[ProgramGroup],
    ) -> Result<(Pipeline, String)> {
        let popt = pipeline_compile_options.build();

        let link_options: sys::OptixPipelineLinkOptions = link_options.into();

        let mut log = [0u8; 4096];
        let mut log_len = log.len();

        let mut raw: sys::OptixPipeline = std::ptr::null_mut();

        let res = unsafe {
            optix_call!(optixPipelineCreate(
                ctx.raw,
                &popt,
                &link_options,
                program_groups.as_ptr() as *const _,
                program_groups.len() as u32,
                log.as_mut_ptr() as *mut i8,
                &mut log_len,
                &mut raw,
            ))
        };

        let log = CStr::from_bytes_with_nul(&log[0..log_len])
            .unwrap()
            .to_string_lossy()
            .into_owned();

        match res {
            Ok(()) => Ok((Pipeline { raw }, log)),
            Err(source) => Err(Error::PipelineCreation { source, log }),
        }
    }
}

impl Drop for Pipeline {
    fn drop(&mut self) {
        unsafe {
            sys::optixPipelineDestroy(self.raw);
        }
    }
}

impl Pipeline {
    /// Sets the stack sizes for a pipeline.
    ///
    /// Users are encouraged to see the programming guide and the
    /// implementations of the helper functions to understand how to
    /// construct the stack sizes based on their particular needs.
    /// If this method is not used, an internal default implementation is used.
    /// The default implementation is correct (but not necessarily optimal) as
    /// long as the maximum depth of call trees of CC and DC programs is at most
    /// 2 and no motion transforms are used.
    /// The maxTraversableGraphDepth responds to the maximal number of
    /// traversables visited when calling trace. Every acceleration structure
    /// and motion transform count as one level of traversal. E.g., for a simple
    /// IAS (instance acceleration structure) -> GAS (geometry acceleration
    /// structure) traversal graph, the maxTraversableGraphDepth is two. For
    /// IAS -> MT (motion transform) -> GAS, the maxTraversableGraphDepth is
    /// three. Note that it does not matter whether a IAS or GAS has motion
    /// or not, it always counts as one. Launching optix with exceptions
    /// turned on (see OPTIX_EXCEPTION_FLAG_TRACE_DEPTH) will throw an
    /// exception if the specified maxTraversableGraphDepth is too small.
    ///
    /// # Arguments
    /// * `direct_callable_stack_size_from_traversable` - The direct stack size
    /// requirement for direct callables invoked from IS or AH
    /// * `direct_callable_stack_size_from_state` - The direct stack size
    /// requirement for direct callables invoked from RG, MS, or CH.
    /// * `continuation_stack_size` - The continuation stack requirement.
    /// * `max_traversable_graph_depth` - The maximum depth of a traversable
    ///   graph
    /// passed to trace
    pub fn set_stack_size(
        &self,
        direct_callable_stack_size_from_traversable: u32,
        direct_callable_stack_size_from_state: u32,
        continuation_stack_size: u32,
        max_traversable_graph_depth: u32,
    ) -> Result<()> {
        unsafe {
            Ok(optix_call!(optixPipelineSetStackSize(
                self.raw,
                direct_callable_stack_size_from_traversable,
                direct_callable_stack_size_from_state,
                continuation_stack_size,
                max_traversable_graph_depth,
            ))?)
        }
    }
}
