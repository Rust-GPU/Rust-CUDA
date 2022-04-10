use crate::llvm::{self, Type};
use rustc_target::spec::{LinkerFlavor, MergeFunctions, PanicStrategy, Target, TargetOptions};

pub const DATA_LAYOUT: &str = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64";
pub const TARGET_TRIPLE: &str = "nvptx64-nvidia-cuda";
pub const POINTER_WIDTH: u32 = 64;

/// The pointer width of the current target
pub(crate) unsafe fn usize_ty(llcx: &'_ llvm::Context) -> &'_ Type {
    llvm::LLVMInt64TypeInContext(llcx)
}

pub fn target() -> Target {
    Target {
        arch: "nvptx".to_string(),
        data_layout: DATA_LAYOUT.to_string(),
        llvm_target: "nvptx64-nvidia-cuda".to_string(),
        pointer_width: 64,

        options: TargetOptions {
            os: "cuda".to_string(),
            vendor: "nvidia".to_string(),
            linker_flavor: LinkerFlavor::PtxLinker,
            // nvvm does all the linking for us, but technically its not a linker
            linker: None,

            cpu: "sm_30".to_string(),

            max_atomic_width: Some(64),

            // Unwinding on CUDA is neither feasible nor useful.
            panic_strategy: PanicStrategy::Abort,

            // Needed to use `dylib` and `bin` crate types and the linker.
            dynamic_linking: true,
            executables: true,

            only_cdylib: true,

            // nvvm does all the work of turning the bitcode into ptx
            obj_is_bitcode: true,

            dll_prefix: "".to_string(),
            dll_suffix: ".ptx".to_string(),
            exe_suffix: ".ptx".to_string(),

            // Disable MergeFunctions LLVM optimisation pass because it can
            // produce kernel functions that call other kernel functions.
            // This behavior is not supported by PTX ISA.
            merge_functions: MergeFunctions::Disabled,

            ..Default::default()
        },
    }
}
