use crate::{context::DeviceContext, error::Error, optix_call, sys};
type Result<T, E = Error> = std::result::Result<T, E>;

use std::cmp::min;
use std::ffi::{CStr, CString};

// Kinda nasty hack to work around the fact taht bindgen generates an i32 for enums on windows,
// but a u32 on linux
#[cfg(windows)]
type OptixEnumBaseType = i32;
#[cfg(unix)]
type OptixEnumBaseType = u32;

#[repr(transparent)]
pub struct Pipeline {
    pub(crate) raw: sys::OptixPipeline,
}

#[repr(C)]
#[derive(Debug, Hash, PartialEq, Copy, Clone, Default)]
pub struct PipelineLinkOptions {
    pub max_trace_depth: u32,
    pub debug_level: CompileDebugLevel,
}

impl From<PipelineLinkOptions> for sys::OptixPipelineLinkOptions {
    fn from(o: PipelineLinkOptions) -> Self {
        sys::OptixPipelineLinkOptions {
            maxTraceDepth: o.max_trace_depth,
            debugLevel: o.debug_level as _,
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
    /// Users are encouraged to see the programming guide and the implementations of the
    /// helper functions to understand how to construct the stack sizes based on their
    /// particular needs. If this method is not used, an internal default implementation
    /// is used. The default implementation is correct (but not necessarily optimal) as
    /// long as the maximum depth of call trees of CC and DC programs is at most 2 and
    /// no motion transforms are used. The maxTraversableGraphDepth responds to the
    /// maximal number of traversables visited when calling trace. Every acceleration
    /// structure and motion transform count as one level of traversal. E.g., for a
    /// simple IAS (instance acceleration structure) -> GAS (geometry acceleration
    /// structure) traversal graph, the maxTraversableGraphDepth is two. For IAS -> MT
    /// (motion transform) -> GAS, the maxTraversableGraphDepth is three. Note that it
    /// does not matter whether a IAS or GAS has motion or not, it always counts as one.
    /// Launching optix with exceptions turned on (see OPTIX_EXCEPTION_FLAG_TRACE_DEPTH)
    /// will throw an exception if the specified maxTraversableGraphDepth is too small.
    ///
    /// # Arguments
    /// * `direct_callable_stack_size_from_traversable` - The direct stack size
    ///   requirement for direct callables invoked from IS or AH
    /// * `direct_callable_stack_size_from_state` - The direct stack size requirement
    ///   for direct callables invoked from RG, MS, or CH.
    /// * `continuation_stack_size` - The continuation stack requirement.
    /// * `max_traversable_graph_depth` - The maximum depth of a traversable graph
    ///   passed to trace.
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

#[repr(transparent)]
pub struct Module {
    pub(crate) raw: sys::OptixModule,
}

/// Module compilation optimization level
#[cfg_attr(windows, repr(i32))]
#[cfg_attr(unix, repr(u32))]
#[derive(Debug, Hash, PartialEq, Copy, Clone, Default)]
pub enum CompileOptimizationLevel {
    #[default]
    Default = sys::OptixCompileOptimizationLevel::OPTIX_COMPILE_OPTIMIZATION_DEFAULT,
    Level0 = sys::OptixCompileOptimizationLevel::OPTIX_COMPILE_OPTIMIZATION_LEVEL_0,
    Level1 = sys::OptixCompileOptimizationLevel::OPTIX_COMPILE_OPTIMIZATION_LEVEL_1,
    Level2 = sys::OptixCompileOptimizationLevel::OPTIX_COMPILE_OPTIMIZATION_LEVEL_2,
    Level3 = sys::OptixCompileOptimizationLevel::OPTIX_COMPILE_OPTIMIZATION_LEVEL_3,
}

/// Module compilation debug level
#[cfg_attr(windows, repr(i32))]
#[cfg_attr(unix, repr(u32))]
#[derive(Debug, Hash, PartialEq, Copy, Clone, Default)]
pub enum CompileDebugLevel {
    #[default]
    None = sys::OptixCompileDebugLevel::OPTIX_COMPILE_DEBUG_LEVEL_NONE,
    LineInfo = sys::OptixCompileDebugLevel::OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO,
    Full = sys::OptixCompileDebugLevel::OPTIX_COMPILE_DEBUG_LEVEL_FULL,
}

cfg_if::cfg_if! {
    if #[cfg(any(feature="optix72", feature="optix73"))] {
        #[repr(C)]
        #[derive(Debug, Hash, PartialEq, Copy, Clone)]
        pub struct ModuleCompileOptions {
            pub max_register_count: i32,
            pub opt_level: CompileOptimizationLevel,
            pub debug_level: CompileDebugLevel,
        }

        impl From<&ModuleCompileOptions> for sys::OptixModuleCompileOptions {
            fn from(o: &ModuleCompileOptions) -> sys::OptixModuleCompileOptions {
                sys::OptixModuleCompileOptions {
                    maxRegisterCount: o.max_register_count,
                    optLevel: o.opt_level as _,
                    debugLevel: o.debug_level as _,
                    boundValues: std::ptr::null(),
                    numBoundValues: 0,
                }
            }
        }
    } else {
        #[repr(C)]
        #[derive(Debug, Hash, PartialEq, Copy, Clone)]
        pub struct ModuleCompileOptions {
            pub max_register_count: i32,
            pub opt_level: CompileOptimizationLevel,
            pub debug_level: CompileDebugLevel,
        }

        impl From<&ModuleCompileOptions> for sys::OptixModuleCompileOptions {
            fn from(o: &ModuleCompileOptions) -> sys::OptixModuleCompileOptions {
                sys::OptixModuleCompileOptions {
                    maxRegisterCount: o.max_register_count,
                    optLevel: o.opt_level as u32,
                    debugLevel: o.debug_level as u32,
                }
            }
        }
    }
}

bitflags::bitflags! {
    #[derive(Default, Hash, Clone, Copy, PartialEq, Eq, Debug)]
    pub struct TraversableGraphFlags: OptixEnumBaseType {
        const ALLOW_ANY = sys::OptixTraversableGraphFlags::OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;
        const ALLOW_SINGLE_GAS = sys::OptixTraversableGraphFlags::OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
        const ALLOW_SINGLE_LEVEL_INSTANCING = sys::OptixTraversableGraphFlags::OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
    }
}

bitflags::bitflags! {
    #[derive(Default, Hash, Clone, Copy, PartialEq, Eq, Debug)]
    pub struct ExceptionFlags: OptixEnumBaseType {
        const NONE = sys::OptixExceptionFlags::OPTIX_EXCEPTION_FLAG_NONE;
        const STACK_OVERFLOW = sys::OptixExceptionFlags::OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
        const TRACE_DEPTH = sys::OptixExceptionFlags::OPTIX_EXCEPTION_FLAG_TRACE_DEPTH;
        const USER = sys::OptixExceptionFlags::OPTIX_EXCEPTION_FLAG_USER;
        const DEBUG = sys::OptixExceptionFlags::OPTIX_EXCEPTION_FLAG_DEBUG;
    }
}

bitflags::bitflags! {
    #[derive(Default, Hash, Clone, Copy, PartialEq, Eq, Debug)]
    pub struct PrimitiveTypeFlags: i32 {
        const DEFAULT = 0;
        const CUSTOM =  sys::OptixPrimitiveTypeFlags_OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM;
        const ROUND_QUADRATIC_BSPLINE = sys::OptixPrimitiveTypeFlags_OPTIX_PRIMITIVE_TYPE_FLAGS_ROUND_QUADRATIC_BSPLINE;
        const ROUND_CUBIC_BSPLINE =  sys::OptixPrimitiveTypeFlags_OPTIX_PRIMITIVE_TYPE_FLAGS_ROUND_CUBIC_BSPLINE;
        const ROUND_LINEAR =  sys::OptixPrimitiveTypeFlags_OPTIX_PRIMITIVE_TYPE_FLAGS_ROUND_LINEAR;
        const TRIANGLE = sys::OptixPrimitiveTypeFlags_OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;
    }
}

#[repr(u32)]
pub enum PrimitiveType {
    RoundQuadraticBspline =
        sys::OptixPrimitiveTypeFlags_OPTIX_PRIMITIVE_TYPE_FLAGS_ROUND_QUADRATIC_BSPLINE as u32,
    RoundCubicBspline =
        sys::OptixPrimitiveTypeFlags_OPTIX_PRIMITIVE_TYPE_FLAGS_ROUND_CUBIC_BSPLINE as u32,
    RoundLinear = sys::OptixPrimitiveTypeFlags_OPTIX_PRIMITIVE_TYPE_FLAGS_ROUND_LINEAR as u32,
    Triangle = sys::OptixPrimitiveTypeFlags_OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE as u32,
}

#[derive(Debug, Hash, PartialEq, Clone, Default)]
pub struct PipelineCompileOptions {
    uses_motion_blur: bool,
    traversable_graph_flags: TraversableGraphFlags,
    num_payload_values: i32,
    num_attribute_values: i32,
    exception_flags: ExceptionFlags,
    pipeline_launch_params_variable_name: Option<CString>,
    primitive_type_flags: PrimitiveTypeFlags,
}

impl PipelineCompileOptions {
    pub fn new() -> PipelineCompileOptions {
        PipelineCompileOptions {
            uses_motion_blur: false,
            traversable_graph_flags: TraversableGraphFlags::ALLOW_ANY,
            num_payload_values: 0,
            num_attribute_values: 0,
            exception_flags: ExceptionFlags::NONE,
            pipeline_launch_params_variable_name: None,
            primitive_type_flags: PrimitiveTypeFlags::DEFAULT,
        }
    }

    pub fn build(&self) -> sys::OptixPipelineCompileOptions {
        cfg_if::cfg_if! {
        if #[cfg(feature="optix73")] {
                sys::OptixPipelineCompileOptions {
                    usesMotionBlur: if self.uses_motion_blur { 1 } else { 0 },
                    traversableGraphFlags: self.traversable_graph_flags.bits() as _,
                    numPayloadValues: self.num_payload_values,
                    numAttributeValues: self.num_attribute_values,
                    exceptionFlags: self.exception_flags.bits() as _,
                    pipelineLaunchParamsVariableName: if let Some(ref name) = self
                        .pipeline_launch_params_variable_name {
                            name.as_ptr()
                        } else {
                            std::ptr::null()
                        },
                    usesPrimitiveTypeFlags: self.primitive_type_flags.bits() as u32,
                    reserved: 0,
                    reserved2: 0,
                }
            } else {
                sys::OptixPipelineCompileOptions {
                    usesMotionBlur: if self.uses_motion_blur { 1 } else { 0 },
                    traversableGraphFlags: self.traversable_graph_flags.bits(),
                    numPayloadValues: self.num_payload_values,
                    numAttributeValues: self.num_attribute_values,
                    exceptionFlags: self.exception_flags.bits(),
                    pipelineLaunchParamsVariableName: if let Some(ref name) = self
                        .pipeline_launch_params_variable_name {
                            name.as_ptr()
                        } else {
                            std::ptr::null()
                        },
                    usesPrimitiveTypeFlags: self.primitive_type_flags.bits() as u32,
                }
            }
        }
    }

    pub fn uses_motion_blur(mut self, umb: bool) -> Self {
        self.uses_motion_blur = umb;
        self
    }

    pub fn traversable_graph_flags(mut self, tgf: TraversableGraphFlags) -> Self {
        self.traversable_graph_flags = tgf;
        self
    }

    pub fn num_payload_values(mut self, npv: i32) -> Self {
        self.num_payload_values = npv;
        self
    }

    pub fn num_attribute_values(mut self, nav: i32) -> Self {
        self.num_attribute_values = nav;
        self
    }

    pub fn exception_flags(mut self, ef: ExceptionFlags) -> Self {
        self.exception_flags = ef;
        self
    }

    pub fn pipeline_launch_params_variable_name(mut self, name: &str) -> Self {
        self.pipeline_launch_params_variable_name = Some(
            CString::new(name).expect("pipeline launch params variable name contains nul bytes"),
        );
        self
    }
}

/// # Creating and destroying `Module`s
impl Module {
    pub fn new(
        ctx: &mut DeviceContext,
        module_compile_options: &ModuleCompileOptions,
        pipeline_compile_options: &PipelineCompileOptions,
        ptx: &str,
    ) -> Result<(Module, String)> {
        let cptx = CString::new(ptx).unwrap();
        let mut log = [0u8; 4096];
        let mut log_len = log.len();

        let mopt = module_compile_options.into();
        let popt = pipeline_compile_options.build();

        let mut raw = std::ptr::null_mut();
        let res = unsafe {
            optix_call!(optixModuleCreateFromPTX(
                ctx.raw,
                &mopt as *const _,
                &popt,
                cptx.as_ptr(),
                cptx.as_bytes().len(),
                log.as_mut_ptr() as *mut i8,
                &mut log_len,
                &mut raw,
            ))
        };

        let log = CStr::from_bytes_with_nul(&log[0..min(log_len, log.len())])
            .unwrap()
            .to_string_lossy()
            .into_owned();

        match res {
            Ok(()) => Ok((Module { raw }, log)),
            Err(source) => Err(Error::ModuleCreation { source, log }),
        }
    }

    /// Returns a module containing the intersection program for the built-in
    /// primitive type specified by the builtinISOptions. This module must be used
    /// as the moduleIS for the OptixProgramGroupHitgroup in any SBT record for
    /// that primitive type.
    pub fn builtin_is_module_get(
        ctx: &mut DeviceContext,
        module_compile_options: &ModuleCompileOptions,
        pipeline_compile_options: &PipelineCompileOptions,
        builtin_is_module_type: PrimitiveType,
        uses_motion_blur: bool,
    ) -> Result<Module> {
        let is_options = sys::OptixBuiltinISOptions {
            builtinISModuleType: builtin_is_module_type as _,
            usesMotionBlur: if uses_motion_blur { 1 } else { 0 },
        };

        let mut raw = std::ptr::null_mut();

        unsafe {
            optix_call!(optixBuiltinISModuleGet(
                ctx.raw,
                module_compile_options as *const _ as *const _,
                pipeline_compile_options as *const _ as *const _,
                &is_options as *const _,
                &mut raw,
            ))
            .map(|_| Module { raw })
            .map_err(Error::from)
        }
    }
}

impl Drop for Module {
    fn drop(&mut self) {
        unsafe {
            sys::optixModuleDestroy(self.raw);
        }
    }
}

#[derive(Clone)]
pub struct ProgramGroupModule<'m> {
    pub module: &'m Module,
    pub entry_function_name: CString,
}

pub enum ProgramGroupDesc<'m> {
    Raygen(ProgramGroupModule<'m>),
    Miss(ProgramGroupModule<'m>),
    Exception(ProgramGroupModule<'m>),
    Hitgroup {
        ch: Option<ProgramGroupModule<'m>>,
        ah: Option<ProgramGroupModule<'m>>,
        is: Option<ProgramGroupModule<'m>>,
    },
    Callables {
        dc: Option<ProgramGroupModule<'m>>,
        cc: Option<ProgramGroupModule<'m>>,
    },
}

impl<'m> ProgramGroupDesc<'m> {
    pub fn raygen(module: &'m Module, entry_function_name: &str) -> ProgramGroupDesc<'m> {
        ProgramGroupDesc::Raygen(ProgramGroupModule {
            module,
            entry_function_name: CString::new(entry_function_name).expect("Invalid string"),
        })
    }

    pub fn miss(module: &'m Module, entry_function_name: &str) -> ProgramGroupDesc<'m> {
        ProgramGroupDesc::Miss(ProgramGroupModule {
            module,
            entry_function_name: CString::new(entry_function_name).expect("Invalid string"),
        })
    }

    pub fn exception(module: &'m Module, entry_function_name: &str) -> ProgramGroupDesc<'m> {
        ProgramGroupDesc::Exception(ProgramGroupModule {
            module,
            entry_function_name: CString::new(entry_function_name).expect("Invalid string"),
        })
    }

    pub fn hitgroup(
        ch: Option<(&'m Module, &str)>,
        ah: Option<(&'m Module, &str)>,
        is: Option<(&'m Module, &str)>,
    ) -> ProgramGroupDesc<'m> {
        ProgramGroupDesc::Hitgroup {
            ch: ch.map(|(module, entry_function_name)| ProgramGroupModule {
                module,
                entry_function_name: CString::new(entry_function_name).expect("Invalid string"),
            }),
            ah: ah.map(|(module, entry_function_name)| ProgramGroupModule {
                module,
                entry_function_name: CString::new(entry_function_name).expect("Invalid string"),
            }),
            is: is.map(|(module, entry_function_name)| ProgramGroupModule {
                module,
                entry_function_name: CString::new(entry_function_name).expect("Invalid string"),
            }),
        }
    }
}

/// A group of programs to be associated with a SBT record.
///
/// Modules can contain more than one program. The program in the module is
/// designated by its entry function name as part of the [ProgramGroupDesc]
/// struct passed to [`ProgramGroup::new()`] and
/// [`ProgramGroup::new_single()`], or specified directly in the
/// case of [`ProgramGroup::raygen()`],
/// [`ProgramGroup::miss()`] and
/// [ProgramGroup::hitgroup()`]
///
/// Four program groups can contain only a single program; only hitgroups can
/// designate up to three programs for the closest-hit, any-hit, and
/// intersection programs.
///
/// Programs from modules can be used in any number of [ProgramGroup] objects.
/// The resulting program groups can be used to fill in any number of
/// SBT records. Program groups can also be used across pipelines as long as the
/// compilation options match.
///
/// A hit group specifies the intersection program used to test whether a ray
/// intersects a primitive, together with the hit shaders to be executed when a
/// ray does intersect the primitive. For built-in primitive types, a built-in
/// intersection program should be obtained from
/// [DeviceContext::builtin_is_module_get()] and used in the hit group. As a
/// special case, the intersection program is not required – and is ignored –
/// for triangle primitives.
///
/// # Safety
/// The lifetime of a module must extend to the lifetime of any
/// ProgramGroup that references that module.
///  FIXME (AL): make this sound by storing module lifetimes here
#[repr(transparent)]
pub struct ProgramGroup {
    pub(crate) raw: sys::OptixProgramGroup,
}

impl ProgramGroup {
    /// Use this information to calculate the total required stack sizes for a
    /// particular call graph of NVIDIA OptiX programs.
    ///
    /// To set the stack sizes for a particular pipeline, use
    /// [Pipeline::set_stack_size()](crate::Pipeline::set_stack_size()).
    pub fn get_stack_size(&self) -> Result<StackSizes> {
        let mut stack_sizes = StackSizes::default();
        unsafe {
            Ok(optix_call!(optixProgramGroupGetStackSize(
                self.raw,
                &mut stack_sizes as *mut _ as *mut _
            ))
            .map(|_| stack_sizes)?)
        }
    }
}

impl PartialEq for ProgramGroup {
    fn eq(&self, rhs: &ProgramGroup) -> bool {
        self.raw == rhs.raw
    }
}

/// # Creating and destroying `ProgramGroup`s
impl ProgramGroup {
    /// Create a [ProgramGroup] for each of the [ProgramGroupDesc] objects in
    /// `desc`.
    pub fn new(
        ctx: &mut DeviceContext,
        desc: &[ProgramGroupDesc],
    ) -> Result<(Vec<ProgramGroup>, String)> {
        cfg_if::cfg_if! {
        if #[cfg(any(feature="optix73"))] {
            let pg_options = sys::OptixProgramGroupOptions { reserved: 0 };
        } else {
            let pg_options = sys::OptixProgramGroupOptions { placeholder: 0 };
        }
        }

        let mut log = [0u8; 4096];
        let mut log_len = log.len();

        let pg_desc: Vec<sys::OptixProgramGroupDesc> = desc.iter().map(|d| d.into()).collect();

        let mut raws = vec![std::ptr::null_mut(); pg_desc.len()];

        let res = unsafe {
            optix_call!(optixProgramGroupCreate(
                ctx.raw,
                pg_desc.as_ptr(),
                pg_desc.len() as u32,
                &pg_options,
                log.as_mut_ptr() as *mut i8,
                &mut log_len,
                raws.as_mut_ptr(),
            ))
        };

        let log = CStr::from_bytes_with_nul(&log[0..log_len])
            .unwrap()
            .to_string_lossy()
            .into_owned();

        match res {
            Ok(()) => Ok((
                raws.iter().map(|raw| ProgramGroup { raw: *raw }).collect(),
                log,
            )),
            Err(source) => Err(Error::ProgramGroupCreation { source, log }),
        }
    }

    /// Create a single [ProgramGroup] specified by `desc`.
    pub fn new_single(
        ctx: &mut DeviceContext,
        desc: &ProgramGroupDesc,
    ) -> Result<(ProgramGroup, String)> {
        cfg_if::cfg_if! {
        if #[cfg(any(feature="optix73"))] {
            let pg_options = sys::OptixProgramGroupOptions { reserved: 0 };
        } else {
            let pg_options = sys::OptixProgramGroupOptions { placeholder: 0 };
        }
        }

        let mut log = [0u8; 4096];
        let mut log_len = log.len();

        let pg_desc: sys::OptixProgramGroupDesc = desc.into();

        let mut raw = std::ptr::null_mut();

        let res = unsafe {
            optix_call!(optixProgramGroupCreate(
                ctx.raw,
                &pg_desc,
                1,
                &pg_options,
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
            Ok(()) => Ok((ProgramGroup { raw }, log)),
            Err(source) => Err(Error::ProgramGroupCreation { source, log }),
        }
    }

    /// Create a raygen [ProgramGroup] from `entry_function_name` in `module`.
    pub fn raygen(
        ctx: &mut DeviceContext,
        module: &Module,
        entry_function_name: &str,
    ) -> Result<ProgramGroup> {
        let desc = ProgramGroupDesc::raygen(module, entry_function_name);
        Ok(ProgramGroup::new_single(ctx, &desc)?.0)
    }

    /// Create a miss [ProgramGroup] from `entry_function_name` in `module`.
    pub fn miss(
        ctx: &mut DeviceContext,
        module: &Module,
        entry_function_name: &str,
    ) -> Result<ProgramGroup> {
        let desc = ProgramGroupDesc::miss(module, entry_function_name);
        Ok(ProgramGroup::new_single(ctx, &desc)?.0)
    }

    /// Create an exception [ProgramGroup] from `entry_function_name` in `module`.
    pub fn exception(
        ctx: &mut DeviceContext,
        module: &Module,
        entry_function_name: &str,
    ) -> Result<ProgramGroup> {
        let desc = ProgramGroupDesc::exception(module, entry_function_name);
        Ok(ProgramGroup::new_single(ctx, &desc)?.0)
    }

    /// Create a hitgroup [ProgramGroup] from any combination of
    /// `(module, entry_function_name)` pairs.
    pub fn hitgroup(
        ctx: &mut DeviceContext,
        closest_hit: Option<(&Module, &str)>,
        any_hit: Option<(&Module, &str)>,
        intersection: Option<(&Module, &str)>,
    ) -> Result<ProgramGroup> {
        let desc = ProgramGroupDesc::hitgroup(closest_hit, any_hit, intersection);
        Ok(ProgramGroup::new_single(ctx, &desc)?.0)
    }
}

impl Drop for ProgramGroup {
    fn drop(&mut self) {
        unsafe {
            sys::optixProgramGroupDestroy(self.raw);
        }
    }
}

impl<'m> From<&ProgramGroupDesc<'m>> for sys::OptixProgramGroupDesc {
    fn from(desc: &ProgramGroupDesc<'m>) -> sys::OptixProgramGroupDesc {
        match &desc {
            ProgramGroupDesc::Raygen(ProgramGroupModule {
                module,
                entry_function_name,
            }) => sys::OptixProgramGroupDesc {
                kind: sys::OptixProgramGroupKind::OPTIX_PROGRAM_GROUP_KIND_RAYGEN,
                __bindgen_anon_1: sys::OptixProgramGroupDesc__bindgen_ty_1 {
                    raygen: sys::OptixProgramGroupSingleModule {
                        module: module.raw,
                        entryFunctionName: entry_function_name.as_ptr(),
                    },
                },
                flags: 0,
            },
            ProgramGroupDesc::Miss(ProgramGroupModule {
                module,
                entry_function_name,
            }) => sys::OptixProgramGroupDesc {
                kind: sys::OptixProgramGroupKind::OPTIX_PROGRAM_GROUP_KIND_MISS,
                __bindgen_anon_1: sys::OptixProgramGroupDesc__bindgen_ty_1 {
                    miss: sys::OptixProgramGroupSingleModule {
                        module: module.raw,
                        entryFunctionName: entry_function_name.as_ptr(),
                    },
                },
                flags: 0,
            },
            ProgramGroupDesc::Exception(ProgramGroupModule {
                module,
                entry_function_name,
            }) => sys::OptixProgramGroupDesc {
                kind: sys::OptixProgramGroupKind::OPTIX_PROGRAM_GROUP_KIND_EXCEPTION,
                __bindgen_anon_1: sys::OptixProgramGroupDesc__bindgen_ty_1 {
                    miss: sys::OptixProgramGroupSingleModule {
                        module: module.raw,
                        entryFunctionName: entry_function_name.as_ptr(),
                    },
                },
                flags: 0,
            },
            ProgramGroupDesc::Hitgroup { ch, ah, is } => {
                let mut efn_ch_ptr = std::ptr::null();
                let mut efn_ah_ptr = std::ptr::null();
                let mut efn_is_ptr = std::ptr::null();

                let module_ch = if let Some(pg_ch) = &ch {
                    efn_ch_ptr = pg_ch.entry_function_name.as_ptr();
                    pg_ch.module.raw
                } else {
                    std::ptr::null_mut()
                };

                let module_ah = if let Some(pg_ah) = &ah {
                    efn_ah_ptr = pg_ah.entry_function_name.as_ptr();
                    pg_ah.module.raw
                } else {
                    std::ptr::null_mut()
                };

                let module_is = if let Some(pg_is) = &is {
                    efn_is_ptr = pg_is.entry_function_name.as_ptr();
                    pg_is.module.raw
                } else {
                    std::ptr::null_mut()
                };

                sys::OptixProgramGroupDesc {
                    kind: sys::OptixProgramGroupKind::OPTIX_PROGRAM_GROUP_KIND_HITGROUP,
                    __bindgen_anon_1: sys::OptixProgramGroupDesc__bindgen_ty_1 {
                        hitgroup: sys::OptixProgramGroupHitgroup {
                            moduleCH: module_ch,
                            entryFunctionNameCH: efn_ch_ptr,
                            moduleAH: module_ah,
                            entryFunctionNameAH: efn_ah_ptr,
                            moduleIS: module_is,
                            entryFunctionNameIS: efn_is_ptr,
                        },
                    },
                    flags: 0,
                }
            }
            ProgramGroupDesc::Callables { dc, cc } => {
                let (module_dc, efn_dc) = if let Some(pg_dc) = &dc {
                    (pg_dc.module.raw, pg_dc.entry_function_name.as_ptr())
                } else {
                    (std::ptr::null_mut(), std::ptr::null())
                };

                let (module_cc, efn_cc) = if let Some(pg_cc) = &cc {
                    (pg_cc.module.raw, pg_cc.entry_function_name.as_ptr())
                } else {
                    (std::ptr::null_mut(), std::ptr::null())
                };

                sys::OptixProgramGroupDesc {
                    kind: sys::OptixProgramGroupKind::OPTIX_PROGRAM_GROUP_KIND_CALLABLES,
                    __bindgen_anon_1: sys::OptixProgramGroupDesc__bindgen_ty_1 {
                        callables: sys::OptixProgramGroupCallables {
                            moduleDC: module_dc,
                            entryFunctionNameDC: efn_dc,
                            moduleCC: module_cc,
                            entryFunctionNameCC: efn_cc,
                        },
                    },
                    flags: 0,
                }
            }
        }
    }
}

#[repr(C)]
#[derive(Debug, Default, Copy, Clone, PartialEq)]
pub struct StackSizes {
    pub css_rg: u32,
    pub css_mg: u32,
    pub css_ch: u32,
    pub css_ah: u32,
    pub css_is: u32,
    pub css_cc: u32,
    pub css_dc: u32,
}
