//! Utility crate for easily building CUDA crates using rustc_codegen_nvvm. Derived from rust-gpu's spirv_builder.

pub use nvvm::*;
use serde::Deserialize;
use std::{
    borrow::Borrow,
    env, fmt,
    path::{Path, PathBuf},
    process::{Command, Stdio},
};

// Add logging support
use tracing::{debug, info, warn, error};

#[derive(Debug)]
#[non_exhaustive]
pub enum CudaBuilderError {
    CratePathDoesntExist(PathBuf),
    FailedToCopyPtxFile(std::io::Error),
    BuildFailed,
}

impl fmt::Display for CudaBuilderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaBuilderError::CratePathDoesntExist(path) => {
                write!(f, "Crate path {} does not exist", path.display())
            }
            CudaBuilderError::BuildFailed => f.write_str("Build failed"),
            CudaBuilderError::FailedToCopyPtxFile(err) => {
                f.write_str(&format!("Failed to copy PTX file: {:?}", err))
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DebugInfo {
    None,
    LineTables,
    // NOTE(RDambrosio016): currently unimplemented because it causes a segfault somewhere in LLVM
    // or libnvvm. Probably the latter.
    // Full,
}

impl DebugInfo {
    fn into_nvvm_and_rustc_options(self) -> (String, String) {
        match self {
            DebugInfo::None => unreachable!(),
            DebugInfo::LineTables => ("-generate-line-info".into(), "-Cdebuginfo=1".into()),
            // DebugInfo::Full => ("-g".into(), "-Cdebuginfo=2".into()),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EmitOption {
    LlvmIr,
    Bitcode,
}

/// A builder for easily compiling Rust GPU crates in build.rs
#[derive(Debug)]
pub struct CudaBuilder {
    path_to_crate: PathBuf,
    /// Whether to compile the gpu crate for release.
    /// `true` by default.
    pub release: bool,
    /// An optional path to copy the final ptx file to.
    pub ptx_file_copy_path: Option<PathBuf>,

    /// Whether to generate debug line number info.
    /// This defaults to `true`, but nothing will be generated
    /// if the gpu crate is built as release.
    pub generate_line_info: bool,
    /// Whether to run libnvvm optimizations. This defaults to `false`
    /// but will be set to `true` if release is specified.
    pub nvvm_opts: bool,
    /// The virtual compute architecture to target for PTX generation. This
    /// dictates how certain things are codegenned and may affect performance
    /// and/or which gpus the code can run on.
    ///
    /// You should generally try to pick an arch that will work with most
    /// GPUs you want your program to work with. Make sure to also
    /// use an appropriate compute arch if you are using recent features
    /// such as tensor cores (which need at least 7.x).
    ///
    /// If you are unsure, either leave this option to default, or pick something around 5.2 to 7.x.
    ///
    /// You can find a list of features supported on each arch and a list of GPUs for every
    /// arch [`here`](https://en.wikipedia.org/wiki/CUDA#Version_features_and_specifications).
    ///
    /// NOTE that this does not necessarily mean that code using a certain capability
    /// will not work on older capabilities. It means that if it uses certain
    /// features it may not work.
    ///
    /// This currently defaults to `6.1`. Which corresponds to Pascal, GPUs such as
    /// the GTX 1030, GTX 1050, GTX 1080, Tesla P40, etc. We default to this because
    /// Maxwell (5.x) will be deprecated in CUDA 12 and we anticipate for that. Moreover,
    /// `6.x` contains support for things like f64 atomic add and half precision float ops.
    pub arch: NvvmArch,
    /// Flush denormal values to zero when performing single-precision floating point operations.
    /// `false` by default.
    pub ftz: bool,
    /// Use a fast approximation for single-precision floating point square root.
    /// `false` by default.
    pub fast_sqrt: bool,
    /// Use a fast approximation for single-precision floating point division.
    /// `false` by default.
    pub fast_div: bool,
    /// Enable FMA (fused multiply-add) contraction.
    /// `true` by default.
    pub fma_contraction: bool,
    /// Whether to emit a certain IR. Emitting LLVM IR is useful to debug any codegen
    /// issues. If you are submitting a bug report try to include the LLVM IR file of
    /// the program that contains the offending function.
    pub emit: Option<EmitOption>,
    /// Indicates to the codegen that the program is being compiled for use in the OptiX hardware raytracing library.
    /// This does a couple of things:
    /// - Aggressively inlines all functions.
    /// - Immediately aborts on panic, not going through the panic handler or panicking machinery.
    /// - sets the `optix` cfg.
    ///
    /// Code compiled with this option should always work under CUDA, but it might not be the most efficient or practical.
    ///
    /// `false` by default.
    pub optix: bool,
    /// Whether to override calls to [`libm`](https://docs.rs/libm/latest/libm/) with calls to libdevice intrinsics.
    ///
    /// Libm is used by no_std crates for functions such as sin, cos, fabs, etc. However, CUDA provides
    /// extremely fast GPU-specific implementations of such functions through `libdevice`. Therefore, the codegen
    /// exposes the option to automatically override any calls to libm functions with calls to libdevice functions.
    /// However, this means the overriden functions are likely to not be deterministic, so if you rely on strict
    /// determinism in things like `rapier`, then it may be helpful to disable such a feature.
    ///
    /// `true` by default.
    pub override_libm: bool,
    /// If `true`, the codegen will attempt to place `static` variables in CUDA's
    /// constant memory, which is fast but limited in size (~64KB total across all
    /// statics). The codegen avoids placing any single item too large, but it does not
    /// track cumulative size. Exceeding the limit may cause `IllegalAddress` runtime
    /// errors (CUDA error code: `700`).
    ///
    /// The default is `false`, which places all statics in global memory. This avoids
    /// such errors but may reduce performance and use more general memory. When set to
    /// `false`, you can still annotate `static` variables with
    /// `#[cuda_std::address_space(constant)]` to place them in constant memory
    /// manually. This option only affects automatic placement.
    ///
    /// Future versions may support smarter placement and user-controlled
    /// packing/spilling strategies.
    pub use_constant_memory_space: bool,
    /// Whether to generate any debug info and what level of info to generate.
    pub debug: DebugInfo,
    /// Additional arguments passed to cargo during `cargo build`.
    pub build_args: Vec<String>,
    /// An optional path where to dump LLVM IR of the final output the codegen will feed to libnvvm. Usually
    /// used for debugging.
    pub final_module_path: Option<PathBuf>,
}

impl CudaBuilder {
    pub fn new(path_to_crate_root: impl AsRef<Path>) -> Self {
        let path_to_crate = path_to_crate_root.as_ref().to_owned();
        
        // LOG: Builder initialization
        info!("Creating CudaBuilder for crate at: {}", path_to_crate.display());
        
        Self {
            path_to_crate,
            release: true,
            ptx_file_copy_path: None,
            generate_line_info: true,
            nvvm_opts: true,
            arch: if cfg!(feature = "nvvm-v19") {
                NvvmArch::Compute120
            } else if cfg!(feature = "nvvm-v7") {
                NvvmArch::default()
            } else {
                panic!("No NVVM version feature enabled. Enable either 'nvvm-v7' or 'nvvm-v19'");
            },
            ftz: false,
            fast_sqrt: false,
            fast_div: false,
            fma_contraction: true,
            emit: None,
            optix: false,
            override_libm: true,
            use_constant_memory_space: false,
            debug: DebugInfo::None,
            build_args: vec![],
            final_module_path: None,
        }
    }

    /// Additional arguments passed to cargo during `cargo build`.
    pub fn build_args(mut self, args: &[impl AsRef<str>]) -> Self {
        self.build_args
            .extend(args.iter().map(|s| s.as_ref().to_owned()));
        debug!("Added build args: {:?}", self.build_args);
        self
    }

    /// Whether to generate any debug info and what level of info to generate.
    pub fn debug(mut self, debug_info: DebugInfo) -> Self {
        debug!("Setting debug info level: {:?}", debug_info);
        self.debug = debug_info;
        self
    }

    /// Whether to compile the gpu crate for release.
    pub fn release(mut self, release: bool) -> Self {
        info!("Setting release mode: {}", release);
        self.release = release;
        self.nvvm_opts = release;
        self
    }

    /// Whether to generate debug line number info.
    /// This defaults to `true`, but nothing will be generated
    /// if the gpu crate is built as release.
    pub fn generate_line_info(mut self, generate_line_info: bool) -> Self {
        debug!("Setting generate_line_info: {}", generate_line_info);
        self.generate_line_info = generate_line_info;
        self
    }

    /// Whether to run libnvvm optimizations. This defaults to `false`
    /// but will be set to `true` if release is specified.
    pub fn nvvm_opts(mut self, nvvm_opts: bool) -> Self {
        debug!("Setting nvvm_opts: {}", nvvm_opts);
        self.nvvm_opts = nvvm_opts;
        self
    }

    /// The virtual compute architecture to target for PTX generation. This
    /// dictates how certain things are codegenned and may affect performance
    /// and/or which gpus the code can run on.
    ///
    /// You should generally try to pick an arch that will work with most
    /// GPUs you want your program to work with. Make sure to also
    /// use an appropriate compute arch if you are using recent features
    /// such as tensor cores (which need at least 7.x).
    ///
    /// If you are unsure, either leave this option to default, or pick something around 5.2 to 7.x.
    ///
    /// You can find a list of features supported on each arch and a list of GPUs for every
    /// arch [`here`](https://en.wikipedia.org/wiki/CUDA#Version_features_and_specifications).
    ///
    /// NOTE that this does not necessarily mean that code using a certain capability
    /// will not work on older capabilities. It means that if it uses certain
    /// features it may not work.
    pub fn arch(mut self, arch: NvvmArch) -> Self {
        info!("Setting target architecture: {:?}", arch);
        self.arch = arch;
        self
    }

    /// Flush denormal values to zero when performing single-precision floating point operations.
    pub fn ftz(mut self, ftz: bool) -> Self {
        debug!("Setting flush-to-zero: {}", ftz);
        self.ftz = ftz;
        self
    }

    /// Use a fast approximation for single-precision floating point square root.
    pub fn fast_sqrt(mut self, fast_sqrt: bool) -> Self {
        debug!("Setting fast sqrt: {}", fast_sqrt);
        self.fast_sqrt = fast_sqrt;
        self
    }

    /// Use a fast approximation for single-precision floating point division.
    pub fn fast_div(mut self, fast_div: bool) -> Self {
        debug!("Setting fast div: {}", fast_div);
        self.fast_div = fast_div;
        self
    }

    /// Enable FMA (fused multiply-add) contraction.
    pub fn fma_contraction(mut self, fma_contraction: bool) -> Self {
        debug!("Setting FMA contraction: {}", fma_contraction);
        self.fma_contraction = fma_contraction;
        self
    }

    /// Emit LLVM IR, the exact same as rustc's `--emit=llvm-ir`.
    pub fn emit_llvm_ir(mut self, emit_llvm_ir: bool) -> Self {
        debug!("Setting emit LLVM IR: {}", emit_llvm_ir);
        self.emit = emit_llvm_ir.then_some(EmitOption::LlvmIr);
        self
    }

    /// Emit LLVM Bitcode, the exact same as rustc's `--emit=llvm-bc`.
    pub fn emit_llvm_bitcode(mut self, emit_llvm_bitcode: bool) -> Self {
        debug!("Setting emit LLVM bitcode: {}", emit_llvm_bitcode);
        self.emit = emit_llvm_bitcode.then_some(EmitOption::Bitcode);
        self
    }

    /// Copy the final ptx file to this location once finished building.
    pub fn copy_to(mut self, path: impl AsRef<Path>) -> Self {
        let copy_path = path.as_ref().to_path_buf();
        info!("Setting PTX copy destination: {}", copy_path.display());
        self.ptx_file_copy_path = Some(copy_path);
        self
    }

    /// Indicates to the codegen that the program is being compiled for use in the OptiX hardware raytracing library.
    /// This does a couple of things:
    /// - Aggressively inlines all functions. (not currently implemented but will be in the future)
    /// - Immediately aborts on panic, not going through the panic handler or panicking machinery.
    /// - sets the `optix` cfg.
    ///
    /// Code compiled with this option should always work under CUDA, but it might not be the most efficient or practical.
    pub fn optix(mut self, optix: bool) -> Self {
        info!("Setting OptiX mode: {}", optix);
        self.optix = optix;
        self
    }

    /// Whether to override calls to [`libm`](https://docs.rs/libm/latest/libm/) with calls to libdevice intrinsics.
    ///
    /// Libm is used by no_std crates for functions such as sin, cos, fabs, etc. However, CUDA provides
    /// extremely fast GPU-specific implementations of such functions through `libdevice`. Therefore, the codegen
    /// exposes the option to automatically override any calls to libm functions with calls to libdevice functions.
    /// However, this means the overriden functions are likely to not be deterministic, so if you rely on strict
    /// determinism in things like `rapier`, then it may be helpful to disable such a feature.
    pub fn override_libm(mut self, override_libm: bool) -> Self {
        debug!("Setting libm override: {}", override_libm);
        self.override_libm = override_libm;
        self
    }

    /// If `true`, the codegen will attempt to place `static` variables in CUDA's
    /// constant memory, which is fast but limited in size (~64KB total across all
    /// statics). The codegen avoids placing any single item too large, but it does not
    /// track cumulative size. Exceeding the limit may cause `IllegalAddress` runtime
    /// errors (CUDA error code: `700`).
    ///
    /// If `false`, all statics are placed in global memory. This avoids such errors but
    /// may reduce performance and use more general memory. You can still annotate
    /// `static` variables with `#[cuda_std::address_space(constant)]` to place them in
    /// constant memory manually as this option only affects automatic placement.
    ///
    /// Future versions may support smarter placement and user-controlled
    /// packing/spilling strategies.
    pub fn use_constant_memory_space(mut self, use_constant_memory_space: bool) -> Self {
        debug!("Setting constant memory space usage: {}", use_constant_memory_space);
        self.use_constant_memory_space = use_constant_memory_space;
        self
    }

    /// An optional path where to dump LLVM IR of the final output the codegen will feed to libnvvm. Usually
    /// used for debugging.
    pub fn final_module_path(mut self, path: impl AsRef<Path>) -> Self {
        let module_path = path.as_ref().to_path_buf();
        debug!("Setting final module path: {}", module_path.display());
        self.final_module_path = Some(module_path);
        self
    }

    /// Runs rustc to build the codegen and codegens the gpu crate, returning the path of the final
    /// ptx file. If [`ptx_file_copy_path`](Self::ptx_file_copy_path) is set, this returns the copied path.
    pub fn build(self) -> Result<PathBuf, CudaBuilderError> {
        info!("Starting CUDA build process");
        debug!("Build configuration: {:?}", self);
        
        println!("cargo:rerun-if-changed={}", self.path_to_crate.display());
        
        let path = invoke_rustc(&self)?;
        info!("Build completed successfully, PTX file generated at: {}", path.display());
        
        if let Some(copy_path) = &self.ptx_file_copy_path {
            info!("Copying PTX file from {} to {}", path.display(), copy_path.display());
            std::fs::copy(&path, copy_path).map_err(|e| {
                error!("Failed to copy PTX file: {}", e);
                CudaBuilderError::FailedToCopyPtxFile(e)
            })?;
            info!("PTX file copied successfully");
            Ok(copy_path.clone())
        } else {
            Ok(path)
        }
    }
}

// https://github.com/rust-lang/cargo/blob/1857880b5124580c4aeb4e8bc5f1198f491d61b1/src/cargo/util/paths.rs#L29-L52
fn dylib_path_envvar() -> &'static str {
    if cfg!(windows) {
        "PATH"
    } else if cfg!(target_os = "macos") {
        "DYLD_FALLBACK_LIBRARY_PATH"
    } else {
        "LD_LIBRARY_PATH"
    }
}

fn dylib_path() -> Vec<PathBuf> {
    match env::var_os(dylib_path_envvar()) {
        Some(var) => env::split_paths(&var).collect(),
        None => Vec::new(),
    }
}

fn find_rustc_codegen_nvvm() -> PathBuf {
    info!("Looking for rustc_codegen_nvvm library");
    
    // Determine which version to look for based on enabled features
    let version_suffix = if cfg!(feature = "nvvm-v19") {
        "_v19"
    } else if cfg!(feature = "nvvm-v7") {
        "_v7"
    } else {
        panic!("No NVVM version feature enabled. Enable either 'nvvm-v7' or 'nvvm-v19'");
    };

    let filename = format!(
        "{}rustc_codegen_nvvm{}{}",
        env::consts::DLL_PREFIX,
        version_suffix,
        env::consts::DLL_SUFFIX
    );
    
    debug!("Searching for library: {}", filename);
    
    let paths = dylib_path();
    debug!("Library search paths: {:?}", paths);
    
    for mut path in paths {
        path.push(&filename);
        debug!("Checking path: {}", path.display());
        if path.is_file() {
            info!("Found rustc_codegen_nvvm at: {}", path.display());
            return path;
        }
    }
    
    panic!("Could not find {} in library path", filename);
}

/// Joins strings together while ensuring none of the strings contain the separator.
fn join_checking_for_separators(strings: Vec<impl Borrow<str>>, sep: &str) -> String {
    for s in &strings {
        let s = s.borrow();
        assert!(
            !s.contains(sep),
            "{:?} may not contain separator {:?}",
            s,
            sep
        );
    }
    strings.join(sep)
}

fn invoke_rustc(builder: &CudaBuilder) -> Result<PathBuf, CudaBuilderError> {
    info!("Invoking rustc for CUDA compilation");
    
    // see https://github.com/EmbarkStudios/rust-gpu/blob/main/crates/spirv-builder/src/lib.rs#L385-L392
    // on what this does
    let rustc_codegen_nvvm = find_rustc_codegen_nvvm();

    let mut rustflags = vec![
        format!("-Zcodegen-backend={}", rustc_codegen_nvvm.display()),
        "-Zcrate-attr=feature(register_tool)".into(),
        "-Zcrate-attr=register_tool(nvvm_internal)".into(),
        "-Zcrate-attr=no_std".into(),
        "-Zsaturating_float_casts=false".into(),
    ];

    if let Some(emit) = &builder.emit {
        let string = match emit {
            EmitOption::LlvmIr => "llvm-ir",
            EmitOption::Bitcode => "llvm-bc",
        };
        rustflags.push(format!("--emit={}", string));
        debug!("Added emit option: {}", string);
    }

    let mut llvm_args = vec![NvvmOption::Arch(builder.arch).to_string()];
    debug!("Base LLVM arg - arch: {}", llvm_args[0]);

    if !builder.nvvm_opts {
        llvm_args.push("-opt=0".to_string());
        debug!("Added LLVM arg: -opt=0");
    }

    if builder.ftz {
        llvm_args.push("-ftz=1".to_string());
        debug!("Added LLVM arg: -ftz=1");
    }

    if builder.fast_sqrt {
        llvm_args.push("-prec-sqrt=0".to_string());
        debug!("Added LLVM arg: -prec-sqrt=0");
    }

    if builder.fast_div {
        llvm_args.push("-prec-div=0".to_string());
        debug!("Added LLVM arg: -prec-div=0");
    }

    if !builder.fma_contraction {
        llvm_args.push("-fma=0".to_string());
        debug!("Added LLVM arg: -fma=0");
    }

    if builder.override_libm {
        llvm_args.push("--override-libm".to_string());
        debug!("Added LLVM arg: --override-libm");
    }

    if let Some(path) = &builder.final_module_path {
        llvm_args.push("--final-module-path".to_string());
        llvm_args.push(path.to_str().unwrap().to_string());
        debug!("Added final module path: {}", path.display());
    }

    if builder.debug != DebugInfo::None {
        let (nvvm_flag, rustc_flag) = builder.debug.into_nvvm_and_rustc_options();
        llvm_args.push(nvvm_flag.clone());
        rustflags.push(rustc_flag.clone());
        debug!("Added debug flags - NVVM: {}, rustc: {}", nvvm_flag, rustc_flag);
    }

    let llvm_args = llvm_args.join(" ");
    if !llvm_args.is_empty() {
        rustflags.push(["-Cllvm-args=", &llvm_args].concat());
        debug!("Final LLVM args: {}", llvm_args);
    }

    debug!("Final rustflags: {:?}", rustflags);

    let mut cargo = Command::new("cargo");
    cargo.args([
        "build",
        "--lib",
        "--message-format=json-render-diagnostics",
        "-Zbuild-std=core,alloc",
        "--target=nvptx64-nvidia-cuda",
    ]);

    cargo.args(&builder.build_args);
    debug!("Added cargo build args: {:?}", builder.build_args);

    if builder.release {
        cargo.arg("--release");
        info!("Building in release mode");
    } else {
        info!("Building in debug mode");
    }

    // TODO(RDambrosio016): Remove this once we can get meaningful error messages in panic to work.
    // for now we enable it to remove some useless indirect calls in the ptx.
    cargo.arg("-Zbuild-std-features=panic_immediate_abort");

    if builder.optix {
        cargo.arg("-Zbuild-std-features=panic_immediate_abort");
        cargo.arg("-Zunstable-options");
        cargo.arg("--config");
        cargo.arg("optix=\"1\"");
        info!("OptiX mode enabled");
    }

    // If we're nested in `cargo` invocation, use a different `--target-dir`,
    // to avoid waiting on the same lock (which effectively dead-locks us).
    // This also helps with e.g. RLS, which uses `--target target/rls`,
    // so we'll have a separate `target/rls/cuda-builder` for it.
    if let (Ok(profile), Some(mut dir)) = (
        env::var("PROFILE"),
        env::var_os("OUT_DIR").map(PathBuf::from),
    ) {
        // Strip `$profile/build/*/out`.
        if dir.ends_with("out")
            && dir.pop()
            && dir.pop()
            && dir.ends_with("build")
            && dir.pop()
            && dir.ends_with(profile)
            && dir.pop()
        {
            let target_dir = dir.join("cuda-builder");
            cargo.arg("--target-dir").arg(&target_dir);
            debug!("Using custom target directory: {}", target_dir.display());
        }
    }

    let arch = format!("{:?}0", builder.arch);
    let cuda_arch = arch.strip_prefix("Compute").unwrap();
    cargo.env("CUDA_ARCH", cuda_arch);
    debug!("Set CUDA_ARCH environment variable: {}", cuda_arch);

    let cargo_encoded_rustflags = join_checking_for_separators(rustflags, "\x1f");

    debug!("Executing cargo command: {:?}", cargo);
    info!("Starting cargo build...");
    
    let build = cargo
        .stderr(Stdio::inherit())
        .current_dir(&builder.path_to_crate)
        .env("CARGO_ENCODED_RUSTFLAGS", cargo_encoded_rustflags)
        .output()
        .expect("failed to execute cargo build");

    // `get_last_artifact` has the side-effect of printing invalid lines, so
    // we do that even in case of an error, to let through any useful messages
    // that ended up on stdout instead of stderr.
    let stdout = String::from_utf8(build.stdout).unwrap();
    debug!("Cargo stdout length: {} bytes", stdout.len());
    
    let artifact = get_last_artifact(&stdout);
    
    if build.status.success() {
        info!("Cargo build completed successfully");
        match artifact {
            Some(path) => {
                info!("Generated PTX artifact: {}", path.display());
                Ok(path)
            }
            None => {
                error!("No PTX artifact found despite successful build - did you forget to mark the crate-type as lib/rlib?");
                Err(CudaBuilderError::BuildFailed)
            }
        }
    } else {
        error!("Cargo build failed with exit code: {:?}", build.status.code());
        Err(CudaBuilderError::BuildFailed)
    }
}

#[derive(Deserialize)]
struct RustcOutput {
    reason: String,
    filenames: Option<Vec<String>>,
}

fn get_last_artifact(out: &str) -> Option<PathBuf> {
    debug!("Parsing cargo output for artifacts");
    
    let artifacts: Vec<_> = out
        .lines()
        .filter_map(|line| match serde_json::from_str::<RustcOutput>(line) {
            Ok(line) => Some(line),
            Err(_) => {
                // Pass through invalid lines
                println!("{}", line);
                None
            }
        })
        .filter(|line| line.reason == "compiler-artifact")
        .collect();
    
    debug!("Found {} compiler artifacts", artifacts.len());
    
    let last = artifacts.into_iter().next_back()?;

    let mut filenames = last
        .filenames
        .unwrap()
        .into_iter()
        .filter(|v| {
            let is_ptx = v.ends_with(".ptx");
            if is_ptx {
                debug!("Found PTX file: {}", v);
            }
            is_ptx
        });
        
    let filename = filenames.next()?;
    assert_eq!(filenames.next(), None, "Crate had multiple .ptx artifacts");
    
    info!("Selected PTX artifact: {}", filename);
    Some(filename.into())
}
