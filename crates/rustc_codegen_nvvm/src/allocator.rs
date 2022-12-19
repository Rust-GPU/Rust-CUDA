use crate::llvm::{self, False, True};
use crate::target;
use crate::LlvmMod;
use libc::c_uint;
use rustc_ast::expand::allocator::{AllocatorKind, AllocatorTy, ALLOCATOR_METHODS};
use rustc_middle::ty::TyCtxt;
use rustc_span::symbol::sym;

// adapted from rustc_codegen_llvm
pub(crate) unsafe fn codegen(
    _tcx: TyCtxt<'_>,
    mods: &mut LlvmMod,
    kind: AllocatorKind,
    alloc_error_handler_kind: AllocatorKind,
) {
    let llcx = &*mods.llcx;
    let llmod = mods.llmod.as_ref().unwrap();
    let usize = target::usize_ty(llcx);
    let i8 = llvm::LLVMInt8TypeInContext(llcx);
    let i8p = llvm::LLVMPointerType(i8, 0);
    let void = llvm::LLVMVoidTypeInContext(llcx);

    let mut used = Vec::new();

    for method in ALLOCATOR_METHODS {
        let mut args = Vec::with_capacity(method.inputs.len());
        for ty in method.inputs.iter() {
            match *ty {
                AllocatorTy::Layout => {
                    args.push(usize); // size
                    args.push(usize); // align
                }
                AllocatorTy::Ptr => args.push(i8p),
                AllocatorTy::Usize => args.push(usize),

                AllocatorTy::ResultPtr | AllocatorTy::Unit => panic!("invalid allocator arg"),
            }
        }
        let output = match method.output {
            AllocatorTy::ResultPtr => Some(i8p),
            AllocatorTy::Unit => None,

            AllocatorTy::Layout | AllocatorTy::Usize | AllocatorTy::Ptr => {
                panic!("invalid allocator output")
            }
        };

        let ty = llvm::LLVMFunctionType(
            output.unwrap_or(void),
            args.as_ptr(),
            args.len() as c_uint,
            False,
        );
        let name = format!("__rust_{}", method.name);
        let llfn = llvm::LLVMRustGetOrInsertFunction(llmod, name.as_ptr().cast(), name.len(), ty);

        used.push(llfn);
        // nvvm doesnt support uwtable so dont try to generate it

        let callee = kind.fn_name(method.name);
        let callee =
            llvm::LLVMRustGetOrInsertFunction(llmod, callee.as_ptr().cast(), callee.len(), ty);
        llvm::LLVMRustSetVisibility(callee, llvm::Visibility::Hidden);

        let llbb = llvm::LLVMAppendBasicBlockInContext(llcx, llfn, "entry\0".as_ptr().cast());

        let llbuilder = llvm::LLVMCreateBuilderInContext(llcx);
        llvm::LLVMPositionBuilderAtEnd(llbuilder, llbb);
        let args = args
            .iter()
            .enumerate()
            .map(|(i, _)| llvm::LLVMGetParam(llfn, i as c_uint))
            .collect::<Vec<_>>();
        let ret =
            llvm::LLVMRustBuildCall(llbuilder, callee, args.as_ptr(), args.len() as c_uint, None);
        llvm::LLVMSetTailCall(ret, True);
        if output.is_some() {
            llvm::LLVMBuildRet(llbuilder, ret);
        } else {
            llvm::LLVMBuildRetVoid(llbuilder);
        }
        llvm::LLVMDisposeBuilder(llbuilder);
    }

    // rust alloc error handler
    let args = [usize, usize]; // size, align

    let ty = llvm::LLVMFunctionType(void, args.as_ptr(), args.len() as c_uint, False);
    let name = "__rust_alloc_error_handler".to_string();
    let llfn = llvm::LLVMRustGetOrInsertFunction(llmod, name.as_ptr().cast(), name.len(), ty);

    used.push(llfn);

    // -> ! DIFlagNoReturn
    llvm::Attribute::NoReturn.apply_llfn(llvm::AttributePlace::Function, llfn);

    let kind = alloc_error_handler_kind;
    let callee = kind.fn_name(sym::oom);
    let callee = llvm::LLVMRustGetOrInsertFunction(llmod, callee.as_ptr().cast(), callee.len(), ty);

    used.push(callee);

    // -> ! DIFlagNoReturn
    llvm::Attribute::NoReturn.apply_llfn(llvm::AttributePlace::Function, callee);
    llvm::LLVMRustSetVisibility(callee, llvm::Visibility::Hidden);

    let llbb = llvm::LLVMAppendBasicBlockInContext(llcx, llfn, "entry\0".as_ptr().cast());

    let llbuilder = llvm::LLVMCreateBuilderInContext(llcx);
    llvm::LLVMPositionBuilderAtEnd(llbuilder, llbb);
    let args = args
        .iter()
        .enumerate()
        .map(|(i, _)| llvm::LLVMGetParam(llfn, i as c_uint))
        .collect::<Vec<_>>();
    let ret = llvm::LLVMRustBuildCall(llbuilder, callee, args.as_ptr(), args.len() as c_uint, None);
    llvm::LLVMSetTailCall(ret, True);
    llvm::LLVMBuildRetVoid(llbuilder);
    llvm::LLVMDisposeBuilder(llbuilder);

    let ptr_ty = llvm::LLVMPointerType(llvm::LLVMInt8TypeInContext(llcx), 0);

    for used in &mut used {
        *used = llvm::LLVMConstBitCast(*used, ptr_ty);
    }

    let section = "llvm.metadata\0".as_ptr().cast();
    let array = llvm::LLVMConstArray(ptr_ty, used.as_ptr(), used.len() as u32);
    let g = llvm::LLVMAddGlobal(
        llmod,
        llvm::LLVMTypeOf(array),
        "llvm.used\0".as_ptr().cast(),
    );
    llvm::LLVMSetInitializer(g, array);
    llvm::LLVMRustSetLinkage(g, llvm::Linkage::AppendingLinkage);
    llvm::LLVMSetSection(g, section);
}
