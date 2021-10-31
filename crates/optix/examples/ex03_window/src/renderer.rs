use anyhow::{Context, Result};
use cust::context::{Context as CuContext, ContextFlags};
use cust::device::{Device, DeviceAttribute};
use cust::memory::{CopyDestination, DBox, DBuffer, DeviceCopy, DevicePointer};
use cust::stream::{Stream, StreamFlags};
use cust::CudaFlags;
use optix::{
    context::DeviceContext,
    module::{
        CompileDebugLevel, CompileOptimizationLevel, ExceptionFlags, Module, ModuleCompileOptions,
        PipelineCompileOptions, TraversableGraphFlags,
    },
    pipeline::{Pipeline, PipelineLinkOptions},
    program_group::ProgramGroupDesc,
    shader_binding_table::{SbtRecord, ShaderBindingTable},
};

use crate::vector::V4f32;

pub struct Renderer {
    launch_params: LaunchParams,
    buf_launch_params: DBox<LaunchParams>,
    sbt: optix::sys::OptixShaderBindingTable,
    buf_raygen: DBuffer<RaygenRecord>,
    buf_hitgroup: DBuffer<HitgroupRecord>,
    buf_miss: DBuffer<MissRecord>,
    pipeline: Pipeline,
    color_buffer: DBuffer<V4f32>,
    ctx: DeviceContext,
    stream: Stream,
    cuda_context: CuContext,
}

impl Renderer {
    pub fn new(width: u32, height: u32) -> Result<Renderer, Box<dyn std::error::Error>> {
        init_optix()?;

        // create CUDA and OptiX contexts
        let device = Device::get_device(0)?;

        let cuda_context =
            CuContext::create_and_push(ContextFlags::SCHED_AUTO | ContextFlags::MAP_HOST, device)?;
        let stream = Stream::new(StreamFlags::DEFAULT, None)?;

        let mut ctx = DeviceContext::new(&cuda_context)?;
        ctx.set_log_callback(|_level, tag, msg| println!("[{}]: {}", tag, msg), 4);

        // create module
        let module_compile_options = ModuleCompileOptions {
            max_register_count: 50,
            opt_level: CompileOptimizationLevel::Default,
            debug_level: CompileDebugLevel::None,
        };

        let pipeline_compile_options = PipelineCompileOptions::new("PARAMS")
            .uses_motion_blur(false)
            .num_attribute_values(2)
            .num_payload_values(2)
            .traversable_graph_flags(TraversableGraphFlags::ALLOW_SINGLE_GAS)
            .exception_flags(ExceptionFlags::NONE);

        let ptx = include_str!(concat!(env!("OUT_DIR"), "/src/ex03_window.ptx"));

        let (module, _log) = Module::new(
            &mut ctx,
            &module_compile_options,
            &pipeline_compile_options,
            ptx,
        )
        .context("Create module")?;

        // create raygen program
        let pgdesc_raygen = ProgramGroupDesc::raygen(&module, "__raygen__renderFrame");

        let (pg_raygen, _log) = ctx.program_group_create(&[pgdesc_raygen])?;

        // create miss program
        let pgdesc_miss = ProgramGroupDesc::miss(&module, "__miss__radiance");

        let (pg_miss, _log) = ctx.program_group_create(&[pgdesc_miss])?;

        let pgdesc_hitgroup = ProgramGroupDesc::hitgroup(
            Some((&module, "__closesthit__radiance")),
            Some((&module, "__anyhit__radiance")),
            None,
        );

        // create hitgroup programs
        let (pg_hitgroup, _log) = ctx.program_group_create(&[pgdesc_hitgroup])?;

        // create pipeline
        let mut program_groups = Vec::new();
        program_groups.extend(pg_raygen.iter().cloned());
        program_groups.extend(pg_miss.iter().cloned());
        program_groups.extend(pg_hitgroup.iter().cloned());

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

        pipeline.set_stack_size(2 * 1024, 2 * 1024, 2 * 1024, 1)?;

        // create SBT
        let rec_raygen: Vec<_> = pg_raygen
            .iter()
            .map(|pg| RaygenRecord::pack(0, pg).expect("failed to pack raygen record"))
            .collect();

        let rec_miss: Vec<_> = pg_miss
            .iter()
            .map(|pg| MissRecord::pack(0, pg).expect("failed to pack miss record"))
            .collect();

        let num_objects = 1;
        let rec_hitgroup: Vec<_> = (0..num_objects)
            .map(|i| {
                let object_type = 0;
                let rec = HitgroupRecord::pack(
                    HitgroupSbtData { object_id: i },
                    &pg_hitgroup[object_type],
                )
                .expect("failed to pack hitgroup record");
                rec
            })
            .collect();

        let mut buf_raygen = DBuffer::from_slice(&rec_raygen)?;
        let mut buf_miss = DBuffer::from_slice(&rec_miss)?;
        let mut buf_hitgroup = DBuffer::from_slice(&rec_hitgroup)?;

        let sbt = ShaderBindingTable::new(&mut buf_raygen)
            .miss(&mut buf_miss)
            .hitgroup(&mut buf_hitgroup)
            .build();

        let mut color_buffer = unsafe { DBuffer::uninitialized(width as usize * height as usize)? };

        let launch_params = LaunchParams {
            frame_id: 0,
            color_buffer: color_buffer.as_device_ptr(),
            fb_size: Point2i {
                x: width as i32,
                y: height as i32,
            },
        };

        let buf_launch_params = DBox::new(&launch_params)?;

        Ok(Renderer {
            ctx,
            cuda_context,
            stream,
            launch_params,
            buf_launch_params,
            buf_raygen,
            buf_hitgroup,
            buf_miss,
            sbt,
            pipeline,
            color_buffer,
        })
    }

    pub fn resize(&mut self, width: u32, height: u32) -> Result<(), Box<dyn std::error::Error>> {
        self.color_buffer = unsafe { DBuffer::uninitialized((width * height) as usize)? };
        self.launch_params.fb_size.x = width as i32;
        self.launch_params.fb_size.y = height as i32;
        self.launch_params.color_buffer = self.color_buffer.as_device_ptr();
        Ok(())
    }

    pub fn render(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        self.buf_launch_params.copy_from(&self.launch_params)?;
        self.launch_params.frame_id += 1;

        unsafe {
            optix::launch(
                &self.pipeline,
                &self.stream,
                &mut self.buf_launch_params,
                &self.sbt,
                self.launch_params.fb_size.x as u32,
                self.launch_params.fb_size.y as u32,
                1,
            )?;
        }

        self.stream.synchronize()?;

        Ok(())
    }

    pub fn download_pixels(&self, slice: &mut [V4f32]) -> Result<(), Box<dyn std::error::Error>> {
        self.color_buffer.copy_to(slice)?;
        Ok(())
    }
}

#[repr(C)]
#[derive(Copy, Clone)]
struct Point2i {
    pub x: i32,
    pub y: i32,
}

#[repr(C)]
#[derive(Copy, Clone)]
struct LaunchParams {
    pub color_buffer: DevicePointer<V4f32>,
    pub fb_size: Point2i,
    pub frame_id: i32,
}

unsafe impl DeviceCopy for LaunchParams {}

type RaygenRecord = SbtRecord<i32>;
type MissRecord = SbtRecord<i32>;

#[derive(Copy, Clone, Default)]
struct HitgroupSbtData {
    object_id: u32,
}
unsafe impl DeviceCopy for HitgroupSbtData {}
type HitgroupRecord = SbtRecord<HitgroupSbtData>;

fn init_optix() -> Result<(), Box<dyn std::error::Error>> {
    cust::init(CudaFlags::empty())?;
    let device_count = Device::num_devices()?;
    if device_count == 0 {
        panic!("No CUDA devices found!");
    }

    optix::init()?;
    Ok(())
}
