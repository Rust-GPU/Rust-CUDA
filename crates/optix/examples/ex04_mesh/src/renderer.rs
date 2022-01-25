use anyhow::{Context, Result};
use cust::context::{Context as CuContext, ContextFlags};
use cust::device::Device;
use cust::memory::{CopyDestination, DeviceBox, DeviceBuffer, DevicePointer, DeviceVariable};
use cust::stream::{Stream, StreamFlags};
use cust::{CudaFlags, DeviceCopy};

use optix::{
    acceleration::IndexedTriangleArray,
    acceleration::{
        Accel, AccelBuildOptions, BuildFlags, GeometryFlags, Traversable, TraversableHandle,
    },
    context::DeviceContext,
    pipeline::{
        CompileDebugLevel, CompileOptimizationLevel, ExceptionFlags, Module, ModuleCompileOptions,
        Pipeline, PipelineCompileOptions, PipelineLinkOptions, ProgramGroup, ProgramGroupDesc,
        TraversableGraphFlags,
    },
    shader_binding_table::{SbtRecord, ShaderBindingTable},
};

use glam::{ivec2, vec3, IVec2, IVec3, Vec3, Vec4};

pub struct Renderer {
    launch_params: DeviceVariable<LaunchParams>,
    sbt: ShaderBindingTable,
    gas: Accel,
    buf_raygen: DeviceBuffer<RaygenRecord>,
    buf_hitgroup: DeviceBuffer<HitgroupRecord>,
    buf_miss: DeviceBuffer<MissRecord>,
    pipeline: Pipeline,
    color_buffer: DeviceBuffer<Vec4>,
    ctx: DeviceContext,
    stream: Stream,
    cuda_context: CuContext,
}

impl Renderer {
    pub fn new(width: u32, height: u32) -> Result<Renderer, Box<dyn std::error::Error>> {
        init_optix()?;

        // create CUDA and OptiX contexts
        let device = Device::get_device(0)?;

        let cuda_context = CuContext::new(device)?;
        let stream = Stream::new(StreamFlags::DEFAULT, None)?;

        let mut ctx = DeviceContext::new(&cuda_context, true)?;
        ctx.set_log_callback(|_level, tag, msg| println!("[{}]: {}", tag, msg), 4)?;

        // create module
        let module_compile_options = ModuleCompileOptions {
            max_register_count: 50,
            opt_level: CompileOptimizationLevel::Default,
            debug_level: CompileDebugLevel::None,
        };

        let pipeline_compile_options = PipelineCompileOptions::new()
            .pipeline_launch_params_variable_name("PARAMS")
            .uses_motion_blur(false)
            .num_attribute_values(2)
            .num_payload_values(2)
            .traversable_graph_flags(TraversableGraphFlags::ALLOW_SINGLE_GAS)
            .exception_flags(ExceptionFlags::NONE);

        let ptx = include_str!("../../resources/ex04_mesh.ptx");

        let (module, _log) = Module::new(
            &mut ctx,
            &module_compile_options,
            &pipeline_compile_options,
            ptx,
        )
        .context("Create module")?;

        // create raygen program
        let pgdesc_raygen = ProgramGroupDesc::raygen(&module, "__raygen__renderFrame");

        let (pg_raygen, _log) = ProgramGroup::new(&mut ctx, &[pgdesc_raygen])?;

        // create miss program
        let pgdesc_miss = ProgramGroupDesc::miss(&module, "__miss__radiance");

        let (pg_miss, _log) = ProgramGroup::new(&mut ctx, &[pgdesc_miss])?;

        let pgdesc_hitgroup = ProgramGroupDesc::hitgroup(
            Some((&module, "__closesthit__radiance")),
            Some((&module, "__anyhit__radiance")),
            None,
        );

        // create hitgroup programs
        let (pg_hitgroup, _log) = ProgramGroup::new(&mut ctx, &[pgdesc_hitgroup])?;

        // create geometry and accels
        let mut vertices = Vec::new();
        let mut indices = Vec::new();
        add_cube(
            vec3(0.0, -1.5, 0.0),
            vec3(10.0, 0.1, 10.0),
            &mut vertices,
            &mut indices,
        );
        add_cube(
            vec3(0.0, 0.0, 0.0),
            vec3(2.0, 2.0, 2.0),
            &mut vertices,
            &mut indices,
        );

        let buf_vertex = DeviceBuffer::from_slice(&vertices)?;
        let buf_indices = DeviceBuffer::from_slice(&indices)?;

        let geometry_flags = GeometryFlags::None;
        let build_inputs =
            IndexedTriangleArray::new(&[&buf_vertex], &buf_indices, &[geometry_flags]);

        let accel_options =
            AccelBuildOptions::new(BuildFlags::PREFER_FAST_TRACE | BuildFlags::ALLOW_COMPACTION);

        // build and compact the GAS
        let gas = Accel::build(&ctx, &stream, &[accel_options], &[build_inputs], true)?;

        stream.synchronize()?;

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

        let mut buf_raygen = DeviceBuffer::from_slice(&rec_raygen)?;
        let mut buf_miss = DeviceBuffer::from_slice(&rec_miss)?;
        let mut buf_hitgroup = DeviceBuffer::from_slice(&rec_hitgroup)?;

        let sbt = ShaderBindingTable::new(&buf_raygen)
            .miss(&buf_miss)
            .hitgroup(&buf_hitgroup);

        // create pipeline
        let mut program_groups = Vec::new();
        program_groups.extend(pg_raygen.into_iter());
        program_groups.extend(pg_miss.into_iter());
        program_groups.extend(pg_hitgroup.into_iter());

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

        let color_buffer =
            unsafe { DeviceBuffer::uninitialized(width as usize * height as usize)? };

        let from = vec3(-10.0, 2.0, -12.0);
        let at = vec3(0.0, 0.0, 0.0);
        let up = vec3(0.0, 1.0, 0.0);

        let cosfovy = 0.66f32;
        let aspect = width as f32 / height as f32;
        let direction = (at - from).normalize();
        let horizontal = cosfovy * aspect * direction.cross(up).normalize();
        let vertical = cosfovy * horizontal.cross(direction).normalize();

        let launch_params = DeviceVariable::new(LaunchParams {
            frame: Frame {
                color_buffer: color_buffer.as_device_ptr(),
                size: ivec2(width as i32, height as i32),
            },
            camera: RenderCamera {
                position: from,
                direction,
                horizontal,
                vertical,
            },
            traversable: gas.handle(),
        })?;

        Ok(Renderer {
            ctx,
            cuda_context,
            stream,
            launch_params,
            gas,
            buf_raygen,
            buf_hitgroup,
            buf_miss,
            sbt,
            pipeline,
            color_buffer,
        })
    }

    pub fn resize(&mut self, width: u32, height: u32) -> Result<(), Box<dyn std::error::Error>> {
        self.color_buffer = unsafe { DeviceBuffer::uninitialized((width * height) as usize)? };
        self.launch_params.frame.size.x = width as i32;
        self.launch_params.frame.size.y = height as i32;
        self.launch_params.frame.color_buffer = self.color_buffer.as_device_ptr();
        Ok(())
    }

    pub fn render(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        self.launch_params.copy_htod()?;

        unsafe {
            optix::launch(
                &self.pipeline,
                &self.stream,
                &self.launch_params,
                &self.sbt,
                self.launch_params.frame.size.x as u32,
                self.launch_params.frame.size.y as u32,
                1,
            )?;
        }

        self.stream.synchronize()?;

        Ok(())
    }

    pub fn download_pixels(&self, slice: &mut [Vec4]) -> Result<(), Box<dyn std::error::Error>> {
        self.color_buffer.copy_to(slice)?;
        Ok(())
    }
}

#[repr(C)]
#[derive(Copy, Clone, DeviceCopy)]
pub struct Frame {
    color_buffer: DevicePointer<Vec4>,
    size: IVec2,
}

#[repr(C)]
#[derive(Copy, Clone, DeviceCopy)]
pub struct RenderCamera {
    position: Vec3,
    direction: Vec3,
    horizontal: Vec3,
    vertical: Vec3,
}

#[repr(C)]
#[derive(Copy, Clone, DeviceCopy)]
pub struct LaunchParams {
    pub frame: Frame,
    pub camera: RenderCamera,
    pub traversable: TraversableHandle,
}

type RaygenRecord = SbtRecord<i32>;
type MissRecord = SbtRecord<i32>;

#[derive(Copy, Clone, Default, DeviceCopy)]
struct HitgroupSbtData {
    object_id: u32,
}
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

pub fn add_cube(center: Vec3, size: Vec3, vertices: &mut Vec<Vec3>, indices: &mut Vec<IVec3>) {
    let start_index = vertices.len() as i32;

    vertices.push((vec3(0.0, 0.0, 0.0)) * size + center - 0.5 * size);
    vertices.push((vec3(1.0, 0.0, 0.0)) * size + center - 0.5 * size);
    vertices.push((vec3(0.0, 1.0, 0.0)) * size + center - 0.5 * size);
    vertices.push((vec3(1.0, 1.0, 0.0)) * size + center - 0.5 * size);
    vertices.push((vec3(0.0, 0.0, 1.0)) * size + center - 0.5 * size);
    vertices.push((vec3(1.0, 0.0, 1.0)) * size + center - 0.5 * size);
    vertices.push((vec3(0.0, 1.0, 1.0)) * size + center - 0.5 * size);
    vertices.push((vec3(1.0, 1.0, 1.0)) * size + center - 0.5 * size);

    const idx: [i32; 36] = [
        0, 1, 3, 2, 3, 0, 5, 7, 6, 5, 6, 4, 0, 4, 5, 0, 5, 1, 2, 3, 7, 2, 7, 6, 1, 5, 6, 1, 7, 3,
        4, 0, 2, 4, 2, 6,
    ];

    for c in idx.chunks(3) {
        indices.push(IVec3::new(
            c[0] + start_index,
            c[1] + start_index,
            c[2] + start_index,
        ));
    }
}
