
# Program Pipeline Creation

# Programming Guide...
<details>
<summary>Click here to expand programming guide</summary>

# Contents
- [Program Input](#program-input)
- [Programming Model](#programming-model)
- [Module Creation](#module-creation)
- [Pipeline Launch Parameter](#pipeline-launch-parameter)
    - [Parameter Specialization](#parameter-specialization)
- [Program Group Creation](#program-group-creation)
- [Pipeline Linking](#pipeline-linking)
- [Pipeline Stack Size](#pipeline-stack-size)
    - [Constructing a Path Tracer](#constructing-a-path-tracer)
- [Compilation Cache](#compilation-cache)

Programs are first compiled into modules of type [`Module`]. One or more modules are combined to create a program group of type [`ProgramGroup`]. Those program groups are then linked into an [`Pipeline`] on the GPU. This is similar to the compile and link process commonly found in software development. The program groups are also used to initialize the header of the SBT record associated with those programs.

The constructors for [`Module`], [`ProgramGroup`], and [`Pipeline`] return a log string. This string is used to report information about any compilation that may have occurred, such as compile errors or verbose information about the compilation result. If an error occurred, the information that would be reported in the log string is also reported by the device context log callback (when provided) (see [`DeviceContext::set_log_callback()`](crate::context::DeviceContext::set_log_callback).

Both mechanisms are provided for these create functions to allow a convenient mechanism for pulling out compilation errors from parallel creation operations without having to determine which output from the logger corresponds to which API invocation.

Symbols in [`Module`] objects may be unresolved and contain extern references to variables and `__device__` functions.

These symbols can be resolved during pipeline creation using the symbols defined in the pipeline modules. Duplicate symbols will trigger an error.

A pipeline contains all programs that are required for a particular ray-tracing launch. An application may use a different pipeline for each launch, or may combine multiple ray-generation programs into a single pipeline.

Most NVIDIA OptiX 7 API functions do not own any significant GPU state; Streaming Assembly (SASS) instructions, which define the executable binary programs in a pipeline, are an exception. The [`Pipeline`] owns the CUDA resource associated with the compiled SASS and it is held until the pipeline is destroyed. This allocation is proportional to the amount of compiled code in the pipeline, typically tens of kilobytes to a few megabytes. However, it is possible to create complex pipelines that require substantially more memory, especially if large static initializers are used. Wherever possible, exercise caution in the number and size of the pipelines.

## Program Input
NVIDIA OptiX 7 programs are encoded in the parallel thread execution instruction set (PTX) language. To create PTX programs, compile CUDA source files using the NVIDIA `nvcc` offline compiler or `nvrtc` JIT compiler. The CUDA code includes PTX device headers used during compilation.

See the `build.rs` files in the examples in this crate for code to compile PTX
as part of the cargo build.

```bash
nvcc -ptx -Ipath-to-optix-sdk/include --use_fast_math myprogram.cu -o myprogram.ptx
```

The nvcc command-line options are explained in more detail as part of the usage description of the compiler options displayed with nvcc --help.
Note the following requirements for nvcc and nvrtc compilation:

* The streaming multiprocessor (SM) target of the input PTX program must be less than or equal to the SM version of the GPU for which the module is compiled.
* To generate code for the minimum supported GPU (Maxwell), use architecture targets for SM 5.0, for example, --gpu-architecture=compute_50. Because OptiX rewrites the code internally, those targets will work on any newer GPU as well.
* CUDA Toolkits 10.2 and newer throw deprecation warnings for SM 5.0 targets. These can be suppressed with the compiler option -Wno-deprecated-gpu-targets.
    If support for Maxwell GPUs is not required, you can use the next higher GPU architecture target SM 6.0 (Pascal) to suppress these warnings.

* Use --machine=64 (-m64). Only 64-bit code is supported in OptiX.
* Define the output type with --ptx. Do not compile to obj or cubin.
* Do not use debug flags -g and -G. OptiX might not handle all debugging instrumentation. This is important when using the Microsoft Visual Studio CUDA integration, which sets these flags as default in the Debug target.
* Enable --relocatable-device-code=true (-rdc). Command nvcc can also use the option --keep-device-functions, which is not supported by nvrtc. These flags prevent the CUDA compiler from eliminating direct or continuation callables as dead code.
* To get smaller and faster code, enable --use_fast_math. This flag enables .approx instructions for trigonometric functions and reciprocals, avoiding inadvertent use of slow double-precision floats. For performance reasons, it is recommended that you set this flag; the only exception is use cases that require more precision.
* To profile your code with Nsight Compute, enable --generate-line-info and set `debug_level = CompileDebugLevel::LineInfo` in both the [`ModuleCompileOptions`] and [`PipelineLinkOptions`] in your application host code.

## Programming Model
The NVIDIA OptiX 7 programming model supports the multiple instruction, multiple data (MIMD) subset of CUDA. Execution must be independent of other threads. For this reason, shared memory usage and warp-wide or block-wide synchronization—such as barriers—are not allowed in the input PTX code. All other GPU instructions are allowed, including math, texture, atomic operations, control flow, and loading data to memory. Special warp-wide instructions like vote and ballot are allowed, but can yield unexpected results as the locality of threads is not guaranteed and neighboring threads can change during execution, unlike in the full CUDA programming model. Still, warp-wide instructions can be used safely when the algorithm in question is independent of locality by, for example, implementing warp-aggregated atomic adds.

The memory model is consistent only within the execution of a single launch index, which starts at the ray-generation invocation and only with subsequent programs reached from any `optixTrace` or callable program. This includes writes to stack allocated variables. Writes from other launch indices may not be available until after the launch is complete. If needed, atomic operations may be used to share data between launch indices, as long as an ordering between launch indices is not required. Memory fences are not supported.

The input PTX should include one or more NVIDIA OptiX 7 programs. The type of program affects how the program can be used during the execution of the pipeline. These program types are specified by prefixing the program's name with the following:

 <table>
 <tr><th>Program Type</th><th>Function Name Prefix</th>
 <tr><td>Ray Generation</td><td><code>__raygen__</code></td></tr>
 <tr><td>Intersection</td><td><code>__intersection__</code></td></tr>
 <tr><td>Any-Hit</td><td><code>__anyhit__</code></td></tr>
 <tr><td>Closest-Hit</td><td><code>__closesthit__</code></td></tr>
 <tr><td>Miss</td><td><code>__miss__</code></td></tr>
 <tr><td>Direct Callable</td><td><code>__direct_callable__</code></td></tr>
 <tr><td>Continuation Callable</td><td><code>__continuation_callable__</code></td></tr>
 <tr><td>Exception</td><td><code>__exception__</code></td></tr>
 </table>

 If a particular function needs to be used with more than one type, then multiple copies with corresponding program prefixes should be generated.

In addition, each program may call a specific set of device-side intrinsics that implement the actual ray-tracing-specific features. (See “Device-side functions”.)

## Module Creation

A module may include multiple programs of any program type. Two option structs control the parameters of the compilation process:

* [`PipelineCompileOptions`] - Must be identical for all modules used to create program groups linked in a single pipeline.
* [`ModuleCompileOptions`] - May vary across the modules within the same pipeline.

These options control general compilation settings, for example, the level of optimization. OptixPipelineCompileOptions controls features of the API such as the usage of custom any-hit programs, curve primitives, motion blur, exceptions, and the number of 32-bit values usable in ray payload and primitive attributes. For example:

```
let module_compile_options = ModuleCompileOptions {
    opt_level: CompileOptimizationLevel::Default,
    debug_level: CompileDebugLevel::LineInfo,
    ..Default::default()
};

let pipeline_compile_options = PipelineCompileOptions::new()
    .uses_motion_blur(false)
    .num_attribute_values(2)
    .num_payload_values(2)
    .pipeline_launch_params_variable_name("PARAMS")
    .exception_flags(ExceptionFlags::NONE)
}
.build();

let (module, log) = Module::new(&ctx, 
    &module_compile_options, 
    &pipeline_compile_options,
    &ptx_string
    )?;
```
The `num_attribute_values` field of [`PipelineCompileOptions`] defines the number of 32-bit words that are reserved to store the attributes. This corresponds to the attribute definition in `optixReportIntersection`. See “Reporting intersections and attribute access”.

<div style = "background-color: #fff7e1; padding: 0">
<span style="float:left; font-size: 4em; padding-left: 0.25em; padding-right: 0.25em;">!</span>
<p style = "padding: 1em">
For best performance when your scene contains nothing but triangles, set uses_primitive_type_flags to PrimitiveTypeFlags::TRIANGLE.
</p>
</div>

## Pipeline Launch Parameter

You specify launch-varying parameters or values that must be accessible from any module through a user-defined variable named in [`PipelineCompileOptions`]. In each module that needs access, declare this variable with `extern` or `extern "C"` linkage and the `__constant__` memory specifier. The size of the variable must match across all modules in a pipeline. Variables of equal size but differing types may trigger undefined behavior.

For example, the following header file defines the variable to share, named PARAMS, as an instance of the Params struct:
```text
struct Params {
    float* image;
    unsigned int image_width;
};

extern "C" __constant__ Params PARAMS;
```

You must match the layout of this struct with an equivalent Rust struct. Take care that CUDA vector types have specific alignment requirements which you must match in the Rust struct or you will trigger invalid memory accesses or undefined behaviour.

```
struct Params {
    image: f32,
    image_width: u32,
}
```

You may also wish to use bindgen to automatically create the equivalent Rust struct from a C/C++ header to ensure they stay in sync.

### Parameter Specialization

Not current implemented

## Program Group Creation
[`ProgramGroup`] objects are created from one to three [`Module`] objects and are used to fill the header of the SBT records. (See [Shader Binding Table](crate::shader_binding_table)) There are five types of program groups: Raygen, Miss, Exception, Hitgroup and Callable.

Modules can contain more than one program. The program in the module is designated by its entry function name as part of the [`ProgramGroupDesc`] struct passed to [`ProgramGroup::new()`](crate::pipeline::ProgramGroup::new). Four program groups can contain only a single program; only the hitgroup program can designate up to three programs for the closest-hit, any-hit, and intersection programs.

Programs from modules can be used in any number of [`ProgramGroup`] objects. The resulting program groups can be used to fill in any number of SBT records. Program groups can also be used across pipelines as long as the compilation options match.

A hit group specifies the intersection program used to test whether a ray intersects a primitive, together with the hit shaders to be executed when a ray does intersect the primitive. For built-in primitive types, a built-in intersection program should be obtained from [`Module::builtin_is_module_get()`](crate::pipeline::Module::builtin_is_module_get) and used in the hit group. As a special case, the intersection program is not required – and is ignored – for triangle primitives.

```
let (module, _log) = Module::new(
    &mut ctx,
    &module_compile_options,
    &pipeline_compile_options,
    ptx,
)?;

let pgdesc_hitgroup = ProgramGroupDesc::hitgroup(
    Some((&module, "__closesthit__radiance")),
    Some((&module, "__anyhit__radiance")),
    None,
);

let (pg_hitgroup, _log) = ProgramGroup::new(&mut ctx, &[pgdesc_hitgroup])?;
```

## Pipeline Linking

After all program groups of a pipeline are defined, they must be linked into an [`Pipeline`]. The resulting [`Pipeline`] object is then used to invoke a ray-generation launch.

When the [`Pipeline`] is linked, some fixed function components may be selected based on [`PipelineLinkOptions`] and [`PipelineCompileOptions`]. These options were previously used to compile the modules in the pipeline. The link options consist of the maximum recursion depth setting for recursive ray tracing, along with pipeline level settings for debugging. However, the value for the maximum recursion depth has an upper limit that overrides an limit set by the link options. (See “Limits”.)

For example, the following code creates and links a [`Pipeline`]:
```
let program_groups = [pg_raygen, pg_miss, pg_hitgroup];

let pipeline_link_options = PipelineLinkOptions {
    max_trace_depth: 2,
    debug_level: CompileDebugLevel::LineInfo,
};

let (pipeline, _log) = Pipeline::new(
    &mut ctx,
    &pipeline_compile_options,
    pipeline_link_options,
    &program_groups,
)?;
```

After [`Pipeline::new()`](crate::pipeline::Pipeline::new) completes, the fully linked module is loaded into the driver.

NVIDIA OptiX 7 uses a small amount of GPU memory per pipeline. This memory is released when the pipeline or device context is destroyed.

## Pipeline Stack Size

The programs in a module may consume two types of stack structure : a direct stack and a continuation stack. The resulting stack needed for launching a pipeline depends on the resulting call graph, so the pipeline must be configured with the appropriate stack size. These sizes can be determined by the compiler for each program group. A pipeline may be reused for different call graphs as long as the set of programs is the same. For this reason, the pipeline stack size is configured separately from the pipeline compilation options.

The direct stack requirements resulting from ray-generation, miss, exception, closest-hit, any-hit and intersection programs and the continuation stack requirements resulting from exception programs are calculated internally and do not need to be configured. The direct stack requirements resulting from direct-callable programs, as well as the continuation stack requirements resulting from ray-generation, miss, closest-hit, any-hit, intersection, and continuation-callable programs need to be configured. If these are not configured explicitly, an internal default implementation is used. When the maximum depth of call trees of continuation-callable and direct-callable programs is two or less, the default implementation is correct (but not necessarily optimal) Even in cases where the default implementation is correct, Users can always provide more precise stack requirements based on their knowledge of a particular call graph structure.

To query individual program groups for their stack requirements, use [`ProgramGroup::get_stack_size`](crate::pipeline::ProgramGroup::get_stack_size). Use this information to calculate the total required stack sizes for a particular call graph of NVIDIA OptiX 7 programs. To set the stack sizes for a particular pipeline, use [`Pipeline::set_stack_size`](crate::pipeline::set_stack_size). For other parameters, helper functions are available to implement these calculations. The following is an explanation about how to compute the stack size for [`Pipeline::set_stack_size()`](crate::pipeline::Pipeline::set_stack_size), starting from a very conservative approach, and refining the estimates step by step.

Let `css_rg` denote the maximum continuation stack size of all ray-generation programs; similarly for miss, closest-hit, any-hit, intersection, and continuation-callable programs. Let `dss_dc` denote the maximum direct stack size of all direct callable programs. Let `max_trace_depth` denote the maximum trace depth (as in [`PipelineLinkOptions::max_trace_depth`](crate::pipeline::PipelineLinkOptions)), and let `max_cc_depth` and `max_dc_depth` denote the maximum depth of call trees of continuation-callable and direct-callable programs, respectively. Then a simple, conservative approach to compute the three parameters of [`Pipeline::set_stack_size`](crate::pipeline::Pipeline::set_stack_size) is:

```
let direct_callable_stack_size_from_traversable = max_dc_depth * dss_dc;
let direct_callable_stack_size_from_state = max_dc_depth * dss_dc;

// Upper bound on continuation stack used by call trees of continuation callables
let css_cc_tree = max_cc_depth * css_cc;

// Upper bound on continuation stack used by closest-hit or miss programs, including 
// the call tree of continuation-callable programs
let css_ch_or_ms_plus_cc_tree = css_ch.max(css_ms) + css_cc_tree;

let continuation_stack_size =
      css_rg
    + css_cc_tree
    + max_trace_depth * css_ch_or_ms_plus_cc_tree
    + css_is
    + css_ah;
```

This computation can be improved in several ways. For the computation of `continuation_stack_size`, the stack sizes `css_is` and `css_ah` are not used on top of the other summands, but can be offset against one level of `css_ch_or_ms_plus_cc_tree`. This gives a more complex but better estimate:

```
let continuation_stack_size =
      css_rg
    + css_cc_tree
    + (max_trace_depth - 1).max(1) * css_ch_or_ms_plus_cc_tree
    + max_trace_depth.min(1) * css_ch_or_ms_plus_cc_tree.max(css_is + css_ah);
```

The computation of the first two terms can be improved if the call trees of direct callable programs are analyzed separately based on the semantic type of their call site. In this context, call sites in any-hit and intersection programs count as traversal, whereas call sites in ray-generation, miss, and closest-hit programs count as state.

```
let direct_callable_stack_size_from_traversable = 
    max_dc_depth_from_traversal * dss_dc_from_traversal;
let direct_callable_stack_size_from_state 
    = max_dc_depth_from_state * dss_dc_from_state;
```

Depending on the scenario, these estimates can be improved further, sometimes substantially. For example, imagine there are two call trees of continuation-callable programs. One call tree is deep, but the involved continuation-callable programs need only a small continuation stack. The other call tree is shallow, but the involved continuation-callable programs needs a quite large continuation stack. The estimate of `css_cc_tree` can be improved as follows:

```
let css_cc_tree = max_cc_depth1 * css_cc1.max(max_cc_depth2 * css_cc2);
```
Similar improvements might be possible for all expressions involving `max_trace_depth` if the ray types are considered separately, for example, camera rays and shadow rays.

### Constructing a Path Tracer

A simple path tracer can be constructed from two ray types: camera rays and shadow rays. The path tracer will consist only of ray-generation, miss, and closest-hit programs, and will not use any-hit, intersection, continuation-callable, or direct-callable programs. The camera rays will invoke only the miss and closest-hit programs `ms1` and `ch1`, respectively. `ch1` might trace shadow rays, which invoke only the miss and closest-hit programs `ms2` and `ch2`, respectively. That is, the maximum trace depth is two and the initial formulas simplify to:

```
let direct_callable_stack_size_from_traversable = max_dc_depth * dss_dc;
let direct_callable_stack_size_from_state = max_dc_depth * dss_dc;
let continuation_stack_size = css_rg + 2 * css_ch1.max(css_ch2).max(css_ms1).max(css_ms2);
```

However, from the call graph structure it is clear that ms2 or ch2 can only be invoked from ch1. This restriction allows for the following estimate:

```
let continuation_stack_size = css_rg + css_ms1.max(css_ch1 + css_ms2.max(css_ch2));
```

This estimate is never worse than the previous one, but often better, for example, in the case where the closest-hit programs have different stack sizes (and the miss programs do not dominate the expression).

## Compilation Cache

Compilation work is triggered automatically when calling [`Module::new()`](crate::pipeline::Module::new) or [`ProgramGroup::new()`](crate::pipeline::ProgramGroup::new), and also potentially during [`Pipeline::new()`](crate::pipeline::Pipeline::new). This work is automatically cached on disk if enabled on the [`DeviceContext`]. Caching reduces compilation effort for recurring programs and program groups. While it is enabled by default, users can disable it through the use of [`DeviceContext::set_cache_enabled()`](crate::context::DeviceContext::set_cache_enabled). See [Context](crate::context) for other options regarding the compilation cache.

Generally, cache entries are compatible with the same driver version and GPU type only.

</details>

 [`DeviceContext`]: crate::context::DeviceContext;
 [`Module`]: crate::pipeline::Module
 [`ProgramGroup`]: crate::pipeline::ProgramGroup
 [`ProgramGroupDesc`]: crate::pipeline::ProgramGroupDesc
 [`Pipeline`]: crate::pipeline::Pipeline
 [`ModuleCompileOptions`]: crate::pipeline::ModuleCompileOptions
 [`PipelineCompileOptions`]: crate::pipeline::PipelineCompileOptions
 [`PipelineLinkOptions`]: crate::pipeline::PipelineLinkOptions

