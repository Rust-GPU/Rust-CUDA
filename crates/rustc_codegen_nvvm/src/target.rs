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
    let mut target_options = TargetOptions::default();
    target_options.os = "cuda".into();
    target_options.vendor = "nvidia".into();
    target_options.linker_flavor = LinkerFlavor::Ptx;
    // nvvm does all the linking for us; but technically its not a linker
    target_options.linker = None;

    target_options.cpu = "sm_30".into();

    target_options.max_atomic_width = Some(64);

    // Unwinding on CUDA is neither feasible nor useful.
    target_options.panic_strategy = PanicStrategy::Abort;

    // Needed to use `dylib` and `bin` crate types and the linker.
    target_options.dynamic_linking = true;
    target_options.executables = true;

    target_options.only_cdylib = true;

    // nvvm does all the work of turning the bitcode into ptx
    target_options.obj_is_bitcode = true;

    target_options.dll_prefix = "".into();
    target_options.dll_suffix = ".ptx".into();
    target_options.exe_suffix = ".ptx".into();

    // Disable MergeFunctions LLVM optimisation pass because it can
    // produce kernel functions that call other kernel functions.
    // This behavior is not supported by PTX ISA.
    target_options.merge_functions = MergeFunctions::Disabled;

    Target {
        arch: "nvptx".into(),
        data_layout: DATA_LAYOUT.into(),
        llvm_target: "nvptx64-nvidia-cuda".into(),
        pointer_width: 64,

        options: target_options,
    }
}
