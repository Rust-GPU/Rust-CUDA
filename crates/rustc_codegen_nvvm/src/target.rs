//! Utility handlers for 32 bit and 64 bit nvptx targets
//!
//! NVVM IR only supports nvptx64-nvidia-cuda and nvptx-nvidia-cuda
//! Therefore we completely ignore the target set in the session.
//! This allows the user to cfg for targets like arm/x86/etc while still
//! compiling for nvptx

use crate::llvm::{self, Type};
use rustc_target::spec::{LinkerFlavor, MergeFunctions, PanicStrategy, Target, TargetOptions};
use std::sync::atomic::{AtomicBool, Ordering};

/// Whether we are compiling for 32 bit (nvptx-nvidia-cuda).
/// This is a global variable so we don't have to pass around a variable to
/// a lot of things when this never varies across codegen invocations.
static TARGET_32_BIT: AtomicBool = AtomicBool::new(false);

/// The data layouts of NVVM targets
/// https://docs.nvidia.com/cuda/nvvm-ir-spec/index.html#data-layout
pub fn data_layout() -> &'static str {
    if TARGET_32_BIT.load(Ordering::SeqCst) {
        "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"
    } else {
        "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"
    }
}

/// The target triples of NVVM targets
/// https://docs.nvidia.com/cuda/nvvm-ir-spec/index.html#target-triple
pub fn target_triple() -> &'static str {
    if TARGET_32_BIT.load(Ordering::SeqCst) {
        "nvptx-nvidia-cuda"
    } else {
        "nvptx64-nvidia-cuda"
    }
}

/// The pointer width of the current target
pub(crate) unsafe fn usize_ty<'ll>(llcx: &'ll llvm::Context) -> &'ll Type {
    if TARGET_32_BIT.load(Ordering::SeqCst) {
        llvm::LLVMInt32TypeInContext(llcx)
    } else {
        llvm::LLVMInt64TypeInContext(llcx)
    }
}

pub fn pointer_size() -> usize {
    if TARGET_32_BIT.load(Ordering::SeqCst) {
        32
    } else {
        64
    }
}

pub fn target() -> Target {
    Target {
        arch: "nvptx".to_string(),
        data_layout: data_layout().to_string(),
        llvm_target: target_triple().to_string(),
        pointer_width: pointer_size() as u32,

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
