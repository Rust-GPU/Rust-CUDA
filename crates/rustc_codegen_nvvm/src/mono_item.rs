use crate::abi::FnAbiLlvmExt;
use crate::attributes;
use crate::attributes::NvvmAttributes;
use crate::consts::linkage_to_llvm;
use crate::context::CodegenCx;
use crate::llvm;
use crate::ty::LayoutLlvmExt;
use rustc_codegen_ssa::traits::*;
use rustc_hir::def_id::{DefId, LOCAL_CRATE};
pub use rustc_middle::mir::mono::MonoItem;
use rustc_middle::mir::mono::{Linkage, Visibility};
use rustc_middle::ty::layout::FnAbiOf;
use rustc_middle::ty::layout::LayoutOf;
use rustc_middle::ty::{self, Instance, TypeFoldable};
use tracing::trace;

pub(crate) fn visibility_to_llvm(linkage: Visibility) -> llvm::Visibility {
    match linkage {
        Visibility::Default => llvm::Visibility::Default,
        Visibility::Hidden => llvm::Visibility::Hidden,
        Visibility::Protected => llvm::Visibility::Protected,
    }
}

impl<'ll, 'tcx> PreDefineMethods<'tcx> for CodegenCx<'ll, 'tcx> {
    fn predefine_static(
        &self,
        def_id: DefId,
        linkage: Linkage,
        visibility: Visibility,
        symbol_name: &str,
    ) {
        trace!("Predefining static with name `{}`", symbol_name);
        let instance = Instance::mono(self.tcx, def_id);
        let ty = instance.ty(self.tcx, ty::ParamEnv::reveal_all());
        let llty = self.layout_of(ty).llvm_type(self);
        let addrspace = self.static_addrspace(instance);

        let g = self
            .define_global(symbol_name, llty, addrspace)
            .unwrap_or_else(|| {
                self.sess().span_fatal(
                    self.tcx.def_span(def_id),
                    &format!("symbol `{}` is already defined", symbol_name),
                )
            });

        unsafe {
            llvm::LLVMRustSetLinkage(g, linkage_to_llvm(linkage));
            llvm::LLVMRustSetVisibility(g, visibility_to_llvm(visibility));
        }

        self.instances.borrow_mut().insert(instance, g);
    }

    fn predefine_fn(
        &self,
        instance: Instance<'tcx>,
        linkage: Linkage,
        visibility: Visibility,
        symbol_name: &str,
    ) {
        trace!(
            "Predefining function with name `{}` with linkage `{:?}` and attributes `{:?}`",
            symbol_name,
            linkage,
            self.tcx.codegen_fn_attrs(instance.def_id())
        );
        assert!(!instance.substs.needs_infer());

        let fn_abi = self.fn_abi_of_instance(instance, ty::List::empty());

        let lldecl = self.declare_fn(symbol_name, fn_abi.llvm_type(self), Some(fn_abi));

        unsafe { llvm::LLVMRustSetLinkage(lldecl, linkage_to_llvm(linkage)) };

        // If we're compiling the compiler-builtins crate, e.g., the equivalent of
        // compiler-rt, then we want to implicitly compile everything with hidden
        // visibility as we're going to link this object all over the place but
        // don't want the symbols to get exported.
        if linkage != Linkage::Internal
            && linkage != Linkage::Private
            && self.tcx.is_compiler_builtins(LOCAL_CRATE)
        {
            unsafe {
                llvm::LLVMRustSetVisibility(lldecl, llvm::Visibility::Hidden);
            }
        } else {
            unsafe {
                llvm::LLVMRustSetVisibility(lldecl, visibility_to_llvm(visibility));
            }
        }

        attributes::from_fn_attrs(self, lldecl, instance);

        let def_id = instance.def_id();
        let attrs = self.tcx.get_attrs(def_id);
        let nvvm_attrs = NvvmAttributes::parse(self, attrs);

        unsafe {
            // if this function is marked as being a kernel, add it
            // to nvvm.annotations per the nvvm ir docs.
            if nvvm_attrs.kernel {
                trace!("Marking function `{:?}` as a kernel", symbol_name);
                let kernel = llvm::LLVMMDStringInContext(self.llcx, "kernel".as_ptr().cast(), 6);
                let mdvals = &[lldecl, kernel, self.const_i32(1)];
                let node =
                    llvm::LLVMMDNodeInContext(self.llcx, mdvals.as_ptr(), mdvals.len() as u32);
                llvm::LLVMAddNamedMetadataOperand(
                    self.llmod,
                    "nvvm.annotations\0".as_ptr().cast(),
                    node,
                );
            }
            if nvvm_attrs.used {
                trace!("Marking function `{:?}` as used", symbol_name);
                let mdvals = &[lldecl];
                let node =
                    llvm::LLVMMDNodeInContext(self.llcx, mdvals.as_ptr(), mdvals.len() as u32);
                llvm::LLVMAddNamedMetadataOperand(
                    self.llmod,
                    "cg_nvvm_used\0".as_ptr().cast(),
                    node,
                );
            }
        }

        self.instances.borrow_mut().insert(instance, lldecl);
    }
}
