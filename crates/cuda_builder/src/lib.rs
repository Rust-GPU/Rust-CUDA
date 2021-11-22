//! Utility crate for easily building CUDA crates using rustc_codegen_nvvm. Derived from rust-gpu's spirv_builder.

pub use nvvm::*;
use serde::Deserialize;
use std::{
    borrow::Borrow,
    env, fmt,
    path::{Path, PathBuf},
    process::{Command, Stdio},
};

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

pub enum EmitOption {
    LlvmIr,
    Bitcode,
}

/// A builder for easily compiling Rust GPU crates in build.rs
pub struct CudaBuilder {
    path_to_crate: PathBuf,
    /// Whether to compile the gpu crate for release.
    /// `true` by default.
    pub release: bool,
    /// Whether to use 32 bit nvptx. Note that this is not tested much, so
    /// it may break in certain cases. You should always use 64 bit nvptx.
    /// `false` by default.
    pub nvptx_32: bool,
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
}

impl CudaBuilder {
    pub fn new(path_to_crate_root: impl AsRef<Path>) -> Self {
        Self {
            path_to_crate: path_to_crate_root.as_ref().to_owned(),
            release: true,
            nvptx_32: false,
            ptx_file_copy_path: None,
            generate_line_info: true,
            nvvm_opts: true,
            arch: NvvmArch::Compute61,
            ftz: false,
            fast_sqrt: false,
            fast_div: false,
            fma_contraction: true,
            emit: None,
            optix: false,
        }
    }

    /// Whether to compile the gpu crate for release.
    pub fn release(mut self, release: bool) -> Self {
        self.release = release;
        self.nvvm_opts = release;
        self
    }

    /// Whether to use 32 bit nvptx. Note that this is not tested much, so
    /// it may break in certain cases. You should always use 64 bit nvptx.
    pub fn nvptx_32(mut self, nvptx_32: bool) -> Self {
        self.nvptx_32 = nvptx_32;
        self
    }

    /// Whether to generate debug line number info.
    /// This defaults to `true`, but nothing will be generated
    /// if the gpu crate is built as release.
    pub fn generate_line_info(mut self, generate_line_info: bool) -> Self {
        self.generate_line_info = generate_line_info;
        self
    }

    /// Whether to run libnvvm optimizations. This defaults to `false`
    /// but will be set to `true` if release is specified.
    pub fn nvvm_opts(mut self, nvvm_opts: bool) -> Self {
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
        self.arch = arch;
        self
    }

    /// Flush denormal values to zero when performing single-precision floating point operations.
    pub fn ftz(mut self, ftz: bool) -> Self {
        self.ftz = ftz;
        self
    }

    /// Use a fast approximation for single-precision floating point square root.
    pub fn fast_sqrt(mut self, fast_sqrt: bool) -> Self {
        self.fast_sqrt = fast_sqrt;
        self
    }

    /// Use a fast approximation for single-precision floating point division.
    pub fn fast_div(mut self, fast_div: bool) -> Self {
        self.fast_div = fast_div;
        self
    }

    /// Enable FMA (fused multiply-add) contraction.
    pub fn fma_contraction(mut self, fma_contraction: bool) -> Self {
        self.fma_contraction = fma_contraction;
        self
    }

    /// Emit LLVM IR, the exact same as rustc's `--emit=llvm-ir`.
    pub fn emit_llvm_ir(mut self, emit_llvm_ir: bool) -> Self {
        self.emit = emit_llvm_ir.then(|| EmitOption::LlvmIr);
        self
    }

    /// Emit LLVM Bitcode, the exact same as rustc's `--emit=llvm-bc`.
    pub fn emit_llvm_bitcode(mut self, emit_llvm_bitcode: bool) -> Self {
        self.emit = emit_llvm_bitcode.then(|| EmitOption::Bitcode);
        self
    }

    /// Copy the final ptx file to this location once finished building.
    pub fn copy_to(mut self, path: impl AsRef<Path>) -> Self {
        self.ptx_file_copy_path = Some(path.as_ref().to_path_buf());
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
        self.optix = optix;
        self
    }

    /// Runs rustc to build the codegen and codegens the gpu crate, returning the path of the final
    /// ptx file. If [`ptx_file_copy_path`](Self::ptx_file_copy_path) is set, this returns the copied path.
    pub fn build(self) -> Result<PathBuf, CudaBuilderError> {
        println!("cargo:rerun-if-changed={}", self.path_to_crate.display());
        let path = invoke_rustc(&self)?;
        if let Some(copy_path) = self.ptx_file_copy_path {
            std::fs::copy(path, &copy_path)
                .map_err(|x| CudaBuilderError::FailedToCopyPtxFile(x))?;
            Ok(copy_path)
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
    let filename = format!(
        "{}rustc_codegen_nvvm{}",
        env::consts::DLL_PREFIX,
        env::consts::DLL_SUFFIX
    );
    for mut path in dylib_path() {
        path.push(&filename);
        if path.is_file() {
            return path;
        }
    }
    panic!("Could not find {} in library path", filename);
}

fn add_libnvvm_to_path() {
    let split_paths = env::var_os(dylib_path_envvar()).unwrap_or_default();
    let mut paths = env::split_paths(&split_paths).collect::<Vec<_>>();
    let libnvvm_path = PathBuf::from(find_cuda_helper::find_cuda_root().unwrap())
        .join("nvvm")
        .join("bin");
    paths.push(libnvvm_path);
    let joined = env::join_paths(&paths).expect("Failed to join paths for PATH");
    env::set_var(dylib_path_envvar(), joined);
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
    // see https://github.com/EmbarkStudios/rust-gpu/blob/main/crates/spirv-builder/src/lib.rs#L385-L392
    // on what this does
    let rustc_codegen_nvvm = find_rustc_codegen_nvvm();

    add_libnvvm_to_path();

    let mut rustflags = vec![format!(
        "-Zcodegen-backend={}",
        rustc_codegen_nvvm.display(),
    )];

    if let Some(emit) = &builder.emit {
        let string = match emit {
            EmitOption::LlvmIr => "llvm-ir",
            EmitOption::Bitcode => "llvm-bc",
        };
        rustflags.push(format!("--emit={}", string));
    }

    let mut llvm_args = vec![];

    llvm_args.push(NvvmOption::Arch(builder.arch).to_string());

    if !builder.nvvm_opts {
        llvm_args.push("-opt=0".to_string());
    }

    if builder.ftz {
        llvm_args.push("-ftz=1".to_string());
    }

    if builder.fast_sqrt {
        llvm_args.push("-prec-sqrt=0".to_string());
    }

    if builder.fast_div {
        llvm_args.push("-prec-div=0".to_string());
    }

    if !builder.fma_contraction {
        llvm_args.push("-fma=0".to_string());
    }

    let llvm_args = llvm_args.join(" ");
    if !llvm_args.is_empty() {
        rustflags.push(["-Cllvm-args=", &llvm_args].concat());
    }

    let target = if builder.nvptx_32 {
        "nvptx-nvidia-cuda"
    } else {
        "nvptx64-nvidia-cuda"
    };

    let mut cargo = Command::new("cargo");
    cargo.args(&[
        "build",
        "--lib",
        "--message-format=json-render-diagnostics",
        "-Zbuild-std=core,alloc",
        "--target",
        target,
    ]);

    if builder.release {
        cargo.arg("--release");
    }

    if builder.optix {
        cargo.arg("-Zbuild-std-features=panic_immediate_abort");
        cargo.arg("-Zunstable-options");
        cargo.arg("--config");
        cargo.arg("optix=\"1\"");
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
            cargo.arg("--target-dir").arg(dir.join("cuda-builder"));
        }
    }

    let cargo_encoded_rustflags = join_checking_for_separators(rustflags, "\x1f");

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
    let artifact = get_last_artifact(&stdout);
    if build.status.success() {
        Ok(artifact.expect("Artifact created when compilation succeeded"))
    } else {
        Err(CudaBuilderError::BuildFailed)
    }
}

#[derive(Deserialize)]
struct RustcOutput {
    reason: String,
    filenames: Option<Vec<String>>,
}

fn get_last_artifact(out: &str) -> Option<PathBuf> {
    let last = out
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
        .last()
        .expect("Did not find output file in rustc output");

    let mut filenames = last
        .filenames
        .unwrap()
        .into_iter()
        .filter(|v| v.ends_with(".ptx"));
    let filename = filenames.next()?;
    assert_eq!(filenames.next(), None, "Crate had multiple .ptx artifacts");
    Some(filename.into())
}
