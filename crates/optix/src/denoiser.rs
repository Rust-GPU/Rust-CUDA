//! GPU-accelerated image denoising using OptiX's AI models.

use std::{
    marker::PhantomData,
    mem::MaybeUninit,
    sync::{Arc, Mutex},
};

use cust::{
    error::CudaResult,
    memory::{DeviceBox, DeviceBuffer, DeviceCopy, DevicePointer, GpuBuffer, UnifiedBuffer},
    prelude::Stream,
};

use crate::{context::DeviceContext, error::Error, optix_call};
type Result<T, E = Error> = std::result::Result<T, E>;

// can't zero initialize, OptixPixelFormat is not zero-initializable.
fn null_optix_image() -> optix_sys::OptixImage2D {
    optix_sys::OptixImage2D {
        data: 0,
        width: 0,
        height: 0,
        pixelStrideInBytes: 0,
        rowStrideInBytes: 0,
        format: optix_sys::OptixPixelFormat::OPTIX_PIXEL_FORMAT_FLOAT2,
    }
}

/// Different kinds of built-in OptiX models suited for different kinds of images.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DenoiserModelKind {
    /// A model suited for low dynamic range input.
    Ldr,
    /// A model suited for high dynamic range input.
    Hdr,
    /// A model suited for high dynamic range input with support for AOVs (Arbitrary Output Variables).
    Aov,
    /// A model suited for high dynamic range input that is temporally stable.
    Temporal,
}

impl DenoiserModelKind {
    /// Converts this model kind to its raw counterpart.
    pub fn to_raw(self) -> optix_sys::OptixDenoiserModelKind::Type {
        match self {
            Self::Ldr => optix_sys::OptixDenoiserModelKind::OPTIX_DENOISER_MODEL_KIND_LDR,
            Self::Hdr => optix_sys::OptixDenoiserModelKind::OPTIX_DENOISER_MODEL_KIND_HDR,
            Self::Aov => optix_sys::OptixDenoiserModelKind::OPTIX_DENOISER_MODEL_KIND_AOV,
            Self::Temporal => optix_sys::OptixDenoiserModelKind::OPTIX_DENOISER_MODEL_KIND_TEMPORAL,
        }
    }
}

/// Options which may be provided to the [`Denoiser`].
/// All fields false by default.
#[non_exhaustive]
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DenoiserOptions {
    /// Guide the inference model using an image of the raw albedo of the scene.
    /// Potentially improving the quality of the output. If this field is set to `true`,
    /// an image must be provided when invoking the Denoiser, otherwise the function will panic.
    pub guide_albedo: bool,
    /// Guide the inference model using an image of the normals of the scene.
    /// Potentially improving the quality of the output. If this field is set to `true`,
    /// an image must be provided when invoking the Denoiser, otherwise the function will panic.
    pub guide_normal: bool,
}

impl DenoiserOptions {
    pub fn to_raw(self) -> optix_sys::OptixDenoiserOptions {
        optix_sys::OptixDenoiserOptions {
            guideAlbedo: self.guide_albedo as u32,
            guideNormal: self.guide_normal as u32,
            denoiseAlpha: optix_sys::OptixDenoiserAlphaMode::OPTIX_DENOISER_ALPHA_MODE_COPY,
        }
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DenoiserSizes {
    pub state_size_in_bytes: usize,
    pub scratch_size_in_bytes_with_overlap: usize,
    pub scratch_size_in_bytes_without_overlap: usize,
    pub overlap_window_size_in_pixels: u32,
}

impl DenoiserSizes {
    pub fn from_raw(raw: optix_sys::OptixDenoiserSizes) -> Self {
        Self {
            state_size_in_bytes: raw.stateSizeInBytes,
            scratch_size_in_bytes_with_overlap: raw.withOverlapScratchSizeInBytes,
            scratch_size_in_bytes_without_overlap: raw.withoutOverlapScratchSizeInBytes,
            overlap_window_size_in_pixels: raw.overlapWindowSizeInPixels,
        }
    }
}

// we keep track of state we allocated to safety-check invocations of the denoiser.
#[derive(Debug)]
struct InternalDenoiserState {
    state: DeviceBuffer<u8>,
    width: u32,
    height: u32,
    _tiled: bool,
    // we handle scratch memory internally currently, so its fine
    // to drop it when we are done.
    scratch: DeviceBuffer<u8>,
}

/// High level wrapper for OptiX's GPU-accelerated AI image denoiser.
#[derive(Debug)]
pub struct Denoiser {
    raw: optix_sys::OptixDenoiser,
    // retain the options and model kind for sanity-checks when invoking
    // the denoiser.
    options: DenoiserOptions,
    kind: DenoiserModelKind,
    state: Arc<Mutex<Option<InternalDenoiserState>>>,
}

impl Drop for Denoiser {
    fn drop(&mut self) {
        unsafe {
            optix_sys::optixDenoiserDestroy(self.raw);
        }
    }
}

impl Denoiser {
    /// Create a new [`Denoiser`] with a model kind and some options.
    pub fn new(
        ctx: &DeviceContext,
        kind: DenoiserModelKind,
        options: DenoiserOptions,
    ) -> Result<Self> {
        let mut raw = MaybeUninit::uninit();
        unsafe {
            let ctx = ctx.raw;
            let raw_kind = kind.to_raw();
            let raw_options = options.to_raw();
            optix_call!(optixDenoiserCreate(
                ctx,
                raw_kind,
                &raw_options as *const _,
                raw.as_mut_ptr()
            ))?;
            Ok(Self {
                raw: raw.assume_init(),
                options,
                kind,
                state: Arc::new(Mutex::new(None)),
            })
        }
    }

    /// Compute the required memory resources for invoking the denoiser on an image of a certain width and height.
    ///
    /// If tiling is being used, `width` and `height` should not contain the overlap size. Tiling requires
    /// extra overlap areas which is why there is scratch memory with and without tiling requirements.
    pub fn required_gpu_memory(&self, width: u32, height: u32) -> Result<DenoiserSizes> {
        let mut sizes = MaybeUninit::uninit();
        unsafe {
            optix_call!(optixDenoiserComputeMemoryResources(
                self.raw,
                width,
                height,
                sizes.as_mut_ptr()
            ))?;
            Ok(DenoiserSizes::from_raw(sizes.assume_init()))
        }
    }

    /// Set up the denoiser state and automatically allocate scratch memory big enough for invocations of `width` and `height` and smaller.
    ///
    /// `width` and `height` are the __max__ expected dimensions of the image layers in a denoiser launch. Launches may be smaller than `width` and `height`, but
    /// they must not be larger, or the launch will panic.
    ///
    /// OptiX requires scratch memory for carrying out denoising, of a certain size obtained with [`Self::required_gpu_memory`]. This method will handle
    /// the management of the scratch memory automatically. It will reuse allocated scratch memory across denoiser invocations, but not across
    /// state setups, calling setup_state will always reallocate the scratch memory.
    ///
    /// If tiling is being used, `tiled` must be true, this will do a couple of things:
    /// - it automatically adds the overlap required to the width and height of the image.
    /// - it uses the overlap version of the scratch size.
    ///
    /// **This will automatically synchronize the stream to prevent common errors**
    pub fn setup_state(
        &mut self,
        stream: &Stream,
        mut width: u32,
        mut height: u32,
        tiled: bool,
    ) -> Result<()> {
        // first, find out how much memory we need to allocate
        let sizes = self.required_gpu_memory(width, height)?;
        let original_width = width;
        let original_height = height;
        // > 'inputWidth' and 'inputHeight' must include overlap on both sides of the image if tiling is being used
        if tiled {
            width += sizes.overlap_window_size_in_pixels * 2;
            height += sizes.overlap_window_size_in_pixels * 2;
        }

        let scratch_size = if tiled {
            sizes.scratch_size_in_bytes_with_overlap
        } else {
            sizes.scratch_size_in_bytes_without_overlap
        };

        // SAFETY: OptiX will write to this and we never read it or expose the buffer.
        let scratch = unsafe { DeviceBuffer::<u8>::uninitialized(scratch_size) }?;

        let state_size = sizes.state_size_in_bytes;

        // SAFETY: OptiX will write into this, its just temporary alloc.
        let state = unsafe { DeviceBuffer::<u8>::uninitialized(state_size) }?;

        unsafe {
            optix_call!(optixDenoiserSetup(
                self.raw,
                stream.as_inner(),
                width,
                height,
                state.as_device_ptr().as_raw(),
                state_size,
                scratch.as_device_ptr().as_raw(),
                scratch_size
            ))?;
        }

        let internal = InternalDenoiserState {
            state,
            width: original_width,
            height: original_height,
            scratch,
            _tiled: tiled,
        };
        *self.state.lock().unwrap() = Some(internal);
        stream.synchronize()?;
        Ok(())
    }

    // TODO(RDambrosio016): provide a way to do the denoising in-place (same input and output buffer).

    /// Invoke the denoiser, writing the output to `out_buffer` (once the stream is synchronized).
    /// The output image will be of the same width and height as the input image
    ///
    /// # Panics
    ///
    /// This method will panic for any of the following reasons:
    /// - The model kind is AOV (use [`invoke_aov`] instead).
    /// - The state was not initialized (use [`setup_state`]).
    /// - The out buffer is not big enough.
    /// - Any of the guide images were not large enough (they were larger than the width and height given in setup_state).
    /// - Guide images that were specified in options or implied by the model kind were not given.
    /// - `input_image` was larger than the dimensions specified in [`setup_state`].
    #[track_caller]
    pub fn invoke<T: DeviceCopy>(
        &self,
        stream: &Stream,
        guide_images: DenoiserGuideImages,
        input_image: Image,
        parameters: DenoiserParams,
        out_buffer: &mut impl GpuBuffer<T>,
    ) -> Result<()> {
        let state_lock = self.state.lock().unwrap();
        let state = state_lock.as_ref().expect(
            "State was not initialized before invoking the denoiser, call Denoiser::setup_state first"
        );
        let state_width = state.width;
        let state_height = state.height;

        assert!(
            state_width >= input_image.width,
            "State was created with an image width of {} but the input image had a width of {}",
            state_width,
            input_image.width
        );

        assert!(
            state_height >= input_image.height,
            "State was created with an image height of {} but the input image had a height of {}",
            state_height,
            input_image.height
        );

        assert_ne!(
            self.kind,
            DenoiserModelKind::Aov,
            "Use Denoiser::invoke_aov for AOV models, not invoke"
        );

        let input_bytes = input_image.bytes_used();
        let buf_len_bytes = out_buffer.len() * std::mem::size_of::<T>();
        assert!(
            buf_len_bytes >= input_bytes as usize,
            "Denoiser out_buffer not large enough, expected at least {} bytes, but found {}",
            input_bytes,
            buf_len_bytes
        );

        if self.options.guide_albedo {
            assert!(
                guide_images.albedo.is_some(),
                "Denoiser was created with guide_albedo but a guide image was not provided during invocation"
            );
        }

        if self.options.guide_normal {
            assert!(
                guide_images.normal.is_some(),
                "Denoiser was created with guide_normal but a guide image was not provided during invocation"
            );
        }

        if self.kind == DenoiserModelKind::Temporal {
            assert!(
                guide_images.flow.is_some(),
                "Denoiser was created with a Temporal model, but a flow image was not provided during invocation"
            );
        }

        let raw_guide = guide_images.clone().to_raw();
        let raw_albedo = raw_guide.albedo;
        let raw_normal = raw_guide.normal;
        let raw_flow = raw_guide.flow;

        // default OptixImage2D is zeroed.
        if raw_albedo.data != 0 {
            assert!(
                raw_albedo.width >= state_width,
                "Albedo guide image's width is too small, expected at least {}, but found {}",
                input_image.width,
                raw_albedo.width
            );
            assert!(
                raw_albedo.height >= state_height,
                "Albedo guide image's height is too small, expected at least {}, but found {}",
                input_image.height,
                raw_albedo.height
            );
        }
        if raw_normal.data != 0 {
            assert!(
                raw_normal.width >= state_width,
                "Normal guide image's width is too small, expected at least {}, but found {}",
                input_image.width,
                raw_normal.width
            );
            assert!(
                raw_normal.height >= state_height,
                "Normal guide image's height is too small, expected at least {}, but found {}",
                input_image.height,
                raw_normal.height
            );
        }
        if raw_flow.data != 0 {
            assert!(
                raw_flow.width >= state_width,
                "Flow guide image's width is too small, expected at least {}, but found {}",
                input_image.width,
                raw_flow.width
            );
            assert!(
                raw_flow.height >= state_height,
                "Flow guide image's height is too small, expected at least {}, but found {}",
                input_image.height,
                raw_flow.height
            );
        }

        let raw_params = parameters.to_raw();

        let mut out = input_image.to_raw();
        out.data = out_buffer.as_device_ptr().as_raw();

        let layer = optix_sys::OptixDenoiserLayer {
            type_: optix_sys::OptixDenoiserAOVType::OPTIX_DENOISER_AOV_TYPE_BEAUTY,
            input: input_image.to_raw(),
            previousOutput: null_optix_image(),
            output: out,
        };

        let cloned = guide_images.to_raw();

        unsafe {
            optix_call!(optixDenoiserInvoke(
                self.raw,
                stream.as_inner(),
                &raw_params as *const _,
                state.state.as_device_ptr().as_raw(),
                state.state.len(),
                &cloned as *const _,
                &layer as *const _,
                1, // num-layers
                0, // offsetX
                0, // offsetY
                state.scratch.as_device_ptr().as_raw(),
                state.scratch.len()
            ))?;
        }

        Ok(())
    }
}

/// Parameters to be given to a single invocation of the denoiser.
#[derive(Debug, Default, Clone, Copy)]
pub struct DenoiserParams<'a> {
    /// Whether to denoise the alpha channel if present.
    pub denoise_alpha: bool,
    /// Average log intensity of the input image. If `None`, then denoised results will not be
    /// optimal for very dark or bright input images.
    pub hdr_intensity: Option<&'a DeviceBox<f32>>,
    /// How much of the denoised image to blend into the final image. If set to `1.0`, then the output
    /// image will be composed of 100% the noisy output. If set to `0.0`, the output will be 100% of the denoised input.
    /// Linearly interpolates for other values.
    pub blend_factor: f32,
    /// Used for AOV models, the average log color of the input image, separate for RGB channels.
    pub hdr_average_color: Option<&'a DeviceBox<[f32; 3]>>,
}

impl DenoiserParams<'_> {
    pub fn to_raw(self) -> optix_sys::OptixDenoiserParams {
        optix_sys::OptixDenoiserParams {
            hdrIntensity: self
                .hdr_intensity
                .map(|x| x.as_device_ptr().as_raw())
                .unwrap_or_default(),
            hdrAverageColor: self
                .hdr_average_color
                .map(|x| x.as_device_ptr().as_raw())
                .unwrap_or_default(),
            blendFactor: self.blend_factor,
            temporalModeUsePreviousLayers: 0,
        }
    }
}

/// Optional guide images to be given to the denoiser.
#[derive(Debug, Default, Clone)]
pub struct DenoiserGuideImages<'a> {
    /// The guide albedo image if guide_albedo was specified in [`DenoiserOptions`].
    /// Ignored if it was not specified in the options.
    pub albedo: Option<Image<'a>>,
    /// The guide normal image if guide_normal was specified in [`DenoiserOptions`].
    /// Ignored if it was not specified in the options.
    ///
    /// If the model kind is Temporal, then this image must be an image of 3d vectors in camera
    /// space. Otherwise, only the X and Y channels will be used.
    pub normal: Option<Image<'a>>,
    /// The guide flow image if the model kind is Temporal. It describes for every pixel,
    /// the flow from the previous to the current frame (a 2d vector in pixel space).
    ///
    /// Will be ignored if the model kind is not Temporal.
    pub flow: Option<Image<'a>>,
}

impl DenoiserGuideImages<'_> {
    pub fn to_raw(self) -> optix_sys::OptixDenoiserGuideLayer {
        optix_sys::OptixDenoiserGuideLayer {
            albedo: self
                .albedo
                .map(|i| i.to_raw())
                .unwrap_or_else(null_optix_image),
            normal: self
                .normal
                .map(|i| i.to_raw())
                .unwrap_or_else(null_optix_image),
            flow: self
                .flow
                .map(|i| i.to_raw())
                .unwrap_or_else(null_optix_image),
            previousOutputInternalGuideLayer: null_optix_image(),
            outputInternalGuideLayer: null_optix_image(),
            flowTrustworthiness: null_optix_image(),
        }
    }
}

/// The possible formats that an image given to OptiX could be.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ImageFormat {
    /// Two 16-bit half values, XY
    Half2,
    /// Three 16-bit half values, RGB
    Half3,
    /// Four 16-bit half values, RGBA
    Half4,
    /// Two 32-bit float values, XY
    Float2,
    /// Three 32-bit float values, RGB
    Float3,
    /// Four 32-bit float values, RGBA
    Float4,
    // uchar images seem to be unsupported by basically every method in OptiX currently,
    // so instead of letting optix return an error, we just don't expose the types.

    // /// Two 8-bit u8 values, RGB
    // Uchar3,
    // /// Four 8-bit u8 values, RGBA
    // Uchar4,
}

impl ImageFormat {
    pub fn to_raw(self) -> optix_sys::OptixPixelFormat::Type {
        use ImageFormat::*;

        match self {
            Half2 => optix_sys::OptixPixelFormat::OPTIX_PIXEL_FORMAT_HALF2,
            Half3 => optix_sys::OptixPixelFormat::OPTIX_PIXEL_FORMAT_HALF3,
            Half4 => optix_sys::OptixPixelFormat::OPTIX_PIXEL_FORMAT_HALF4,
            Float2 => optix_sys::OptixPixelFormat::OPTIX_PIXEL_FORMAT_FLOAT2,
            Float3 => optix_sys::OptixPixelFormat::OPTIX_PIXEL_FORMAT_FLOAT3,
            Float4 => optix_sys::OptixPixelFormat::OPTIX_PIXEL_FORMAT_FLOAT4,
            // Uchar3 => optix_sys::OptixPixelFormat::OPTIX_PIXEL_FORMAT_UCHAR3,
            // Uchar4 => optix_sys::OptixPixelFormat::OPTIX_PIXEL_FORMAT_UCHAR4,
        }
    }

    pub fn byte_size(self) -> u32 {
        match self {
            Self::Half2 => 4,
            Self::Half3 => 6,
            Self::Half4 | Self::Float2 => 8,
            Self::Float3 => 12,
            Self::Float4 => 16,
            // Self::Uchar3 => 3,
            // Self::Uchar4 => 4,
        }
    }
}

/// A borrowed GPU buffer interpreted as an image with a certain width, height, and format.
#[derive(Debug, Clone, PartialEq)]
pub struct Image<'a> {
    // TODO(RDambrosio016): maybe change this to a boxed GpuBuffer?
    buffer: DevicePointer<u8>,
    // store
    buffer_size: usize,
    format: ImageFormat,
    width: u32,
    height: u32,
    _phantom: PhantomData<&'a ()>,
}

impl<'a> Image<'a> {
    /// Creates a new image from a specific format, checking the byte buffer length to make sure
    /// it is in bounds.
    ///
    /// # Panics
    ///
    /// Panics in case the byte buffer is not large enough, the buffer
    /// must at least be `(width * height) * format.byte_size()` bytes large.
    #[track_caller]
    pub fn new<T: DeviceCopy>(
        buffer: &'a impl GpuBuffer<T>,
        format: ImageFormat,
        width: u32,
        height: u32,
    ) -> Self {
        Self::validate_buf(buffer, format, width, height);

        Self {
            buffer:
                // SAFETY: this buffer is never written to for the duration of this image being alive.
                // And we know the buffer is large enough to be reinterpreted as a buffer of bytes.
                DevicePointer::from_raw(buffer.as_device_ptr().as_raw()),
            buffer_size: buffer.len() * std::mem::size_of::<T>(),
            format,
            width,
            height,
            _phantom: PhantomData,
        }
    }

    fn validate_buf<T: DeviceCopy>(
        bytes: &impl GpuBuffer<T>,
        format: ImageFormat,
        width: u32,
        height: u32,
    ) {
        let required_size = Self::required_buffer_size(format, width, height);
        let buf_size = std::mem::size_of::<T>() * bytes.len();

        assert!(
            buf_size >= required_size,
            "Buffer for {}x{} {:?} image is not large enough, expected {} bytes, found {}",
            width,
            height,
            format,
            required_size,
            bytes.len()
        );
    }

    /// Allocates a new [`UnifiedBuffer`] for an image of a certain width, height, and format, using
    /// `T::default()` to fill the buffer (so it is not uninitialized).
    ///
    /// If the required bytes are not evenly divisible by `size_of::<T>()`, the resulting buffer size
    /// will be rounded up. For example, if `4` bytes are needed but the size of T is `6`, the buffer's
    /// len will be `1` with a total byte size of `6`.
    pub fn unified_buffer_for_image<T: DeviceCopy + Default>(
        format: ImageFormat,
        width: u32,
        height: u32,
    ) -> CudaResult<UnifiedBuffer<T>> {
        let required_bytes = Self::required_buffer_size(format, width, height);
        let t_size = std::mem::size_of::<T>();
        // round-up division
        let buf_size = required_bytes.div_ceil(t_size);

        UnifiedBuffer::new(&T::default(), buf_size)
    }

    /// The amount of memory in bytes required for an image of a certain width, format, and height
    pub fn required_buffer_size(format: ImageFormat, width: u32, height: u32) -> usize {
        ((width * height) * format.byte_size()) as usize
    }

    pub fn row_stride_in_bytes(&self) -> u32 {
        self.width * self.format.byte_size()
    }

    pub fn pixel_stride_in_bytes(&self) -> u32 {
        self.format.byte_size()
    }

    pub fn to_raw(&self) -> optix_sys::OptixImage2D {
        optix_sys::OptixImage2D {
            width: self.width,
            height: self.height,
            rowStrideInBytes: self.row_stride_in_bytes(),
            pixelStrideInBytes: self.pixel_stride_in_bytes(),
            format: self.format.to_raw(),
            data: self.buffer.as_raw(),
        }
    }

    /// Whether this image can be reinterpreted into an image with a different format and dimensions.
    pub fn can_be_reinterpreted_to(&self, to: ImageFormat, width: u32, height: u32) -> bool {
        let required = Self::required_buffer_size(to, width, height);
        self.buffer_size >= required
    }

    /// Reinterprets this image as an image of a separate format, width, and height. Panics if
    /// the buffer is not large enough.
    #[track_caller]
    pub fn reinterpret(&mut self, to: ImageFormat, width: u32, height: u32) {
        assert!(
            self.can_be_reinterpreted_to(to, width, height),
            "Image cannot be reinterpreted to {}x{} {:?}, buffer is not large enough, expected {} bytes, found {}",
            width,
            height,
            to,
            Self::required_buffer_size(to, width, height),
            self.buffer_size
        );
        self.format = to;
        self.width = width;
        self.height = height;
    }

    pub fn format(&self) -> ImageFormat {
        self.format
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    pub(crate) fn bytes_used(&self) -> u32 {
        self.format.byte_size() * (self.width * self.height)
    }
}
