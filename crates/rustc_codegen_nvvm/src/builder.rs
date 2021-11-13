use crate::context::CodegenCx;
use crate::int_replace::{get_transformed_type, transmute_llval};
use crate::llvm::{self, BasicBlock, LLVMRustGetValueType, Type, Value};
use crate::ty::LayoutLlvmExt;
use libc::{c_char, c_uint};
use rustc_codegen_ssa::common::{AtomicOrdering, IntPredicate, RealPredicate, TypeKind};
use rustc_codegen_ssa::mir::operand::{OperandRef, OperandValue};
use rustc_codegen_ssa::mir::place::PlaceRef;
use rustc_codegen_ssa::traits::*;
use rustc_codegen_ssa::MemFlags;
use rustc_hir::def_id::DefId;
use rustc_middle::ty::layout::{
    FnAbiError, FnAbiOfHelpers, FnAbiRequest, LayoutError, LayoutOfHelpers, TyAndLayout,
};
use rustc_middle::ty::{self, Ty, TyCtxt};
use rustc_span::Span;
use rustc_target::abi::call::FnAbi;
use rustc_target::abi::{self, Align, Size, WrappingRange};
use rustc_target::spec::{HasTargetSpec, Target};
use std::borrow::Cow;
use std::ffi::{CStr, CString};
use std::ops::Deref;
use std::ptr;
use std::sync::Mutex;
use tracing::trace;

// All Builders must have an llfn associated with them
#[must_use]
pub(crate) struct Builder<'a, 'll, 'tcx> {
    pub llbuilder: Mutex<&'ll mut llvm::Builder<'ll>>,
    pub cx: &'a CodegenCx<'ll, 'tcx>,
}

impl<'ll, 'tcx, 'a> Drop for Builder<'a, 'll, 'tcx> {
    fn drop(&mut self) {
        unsafe {
            llvm::LLVMDisposeBuilder(&mut *(&mut **self.llbuilder.lock().unwrap() as *mut _));
        }
    }
}

// cant make this a const because its not a const fn on stable yet :(
fn empty_cstr() -> &'static CStr {
    unsafe { CStr::from_bytes_with_nul_unchecked(b"\0") }
}

/// Empty string, to be used where LLVM expects an instruction name, indicating
/// that the instruction is to be left unnamed (i.e. numbered, in textual IR).
pub(crate) fn unnamed() -> *const c_char {
    empty_cstr().as_ptr()
}

impl<'ll, 'tcx> BackendTypes for Builder<'_, 'll, 'tcx> {
    type Value = <CodegenCx<'ll, 'tcx> as BackendTypes>::Value;
    type Function = <CodegenCx<'ll, 'tcx> as BackendTypes>::Function;
    type BasicBlock = <CodegenCx<'ll, 'tcx> as BackendTypes>::BasicBlock;
    type Type = <CodegenCx<'ll, 'tcx> as BackendTypes>::Type;
    type Funclet = <CodegenCx<'ll, 'tcx> as BackendTypes>::Funclet;

    type DIScope = <CodegenCx<'ll, 'tcx> as BackendTypes>::DIScope;
    type DILocation = <CodegenCx<'ll, 'tcx> as BackendTypes>::DILocation;
    type DIVariable = <CodegenCx<'ll, 'tcx> as BackendTypes>::DIVariable;
}

impl abi::HasDataLayout for Builder<'_, '_, '_> {
    fn data_layout(&self) -> &abi::TargetDataLayout {
        self.cx.data_layout()
    }
}

impl<'tcx> ty::layout::HasTyCtxt<'tcx> for Builder<'_, '_, 'tcx> {
    fn tcx(&self) -> TyCtxt<'tcx> {
        self.cx.tcx
    }
}

impl<'tcx> ty::layout::HasParamEnv<'tcx> for Builder<'_, '_, 'tcx> {
    fn param_env(&self) -> ty::ParamEnv<'tcx> {
        self.cx.param_env()
    }
}

impl<'tcx> HasTargetSpec for Builder<'_, '_, 'tcx> {
    fn target_spec(&self) -> &Target {
        &self.cx.target_spec()
    }
}

impl<'tcx> LayoutOfHelpers<'tcx> for Builder<'_, '_, 'tcx> {
    type LayoutOfResult = TyAndLayout<'tcx>;

    #[inline]
    fn handle_layout_err(&self, err: LayoutError<'tcx>, span: Span, ty: Ty<'tcx>) -> ! {
        self.cx.handle_layout_err(err, span, ty)
    }
}

impl<'ll, 'tcx> FnAbiOfHelpers<'tcx> for Builder<'_, 'll, 'tcx> {
    type FnAbiOfResult = &'tcx FnAbi<'tcx, Ty<'tcx>>;

    #[inline]
    fn handle_fn_abi_err(
        &self,
        err: FnAbiError<'tcx>,
        span: Span,
        fn_abi_request: FnAbiRequest<'tcx>,
    ) -> ! {
        self.cx.handle_fn_abi_err(err, span, fn_abi_request)
    }
}

impl<'ll, 'tcx> Deref for Builder<'_, 'll, 'tcx> {
    type Target = CodegenCx<'ll, 'tcx>;

    #[inline]
    fn deref(&self) -> &Self::Target {
        self.cx
    }
}

impl<'ll, 'tcx> HasCodegen<'tcx> for Builder<'_, 'll, 'tcx> {
    type CodegenCx = CodegenCx<'ll, 'tcx>;
}

macro_rules! builder_methods_for_value_instructions {
    ($($name:ident($($arg:ident),*) => $llvm_capi:ident),+ $(,)?) => {
        $(fn $name(&mut self, $($arg: &'ll Value),*) -> &'ll Value {
            unsafe {
                trace!("binary expr: {:?} with args {:?}", stringify!($name), [$($arg),*]);
                llvm::$llvm_capi(&mut self.llbuilder.lock().unwrap(), $($arg,)* unnamed())
            }
        })+
    }
}

impl<'a, 'll, 'tcx> CoverageInfoBuilderMethods<'tcx> for Builder<'a, 'll, 'tcx> {
    fn set_function_source_hash(
        &mut self,
        _instance: ty::Instance<'tcx>,
        _function_source_hash: u64,
    ) -> bool {
        todo!()
    }

    fn add_coverage_counter(
        &mut self,
        _instance: ty::Instance<'tcx>,
        _index: rustc_middle::mir::coverage::CounterValueReference,
        _region: rustc_middle::mir::coverage::CodeRegion,
    ) -> bool {
        todo!()
    }

    fn add_coverage_counter_expression(
        &mut self,
        _instance: ty::Instance<'tcx>,
        _id: rustc_middle::mir::coverage::InjectedExpressionId,
        _lhs: rustc_middle::mir::coverage::ExpressionOperandId,
        _op: rustc_middle::mir::coverage::Op,
        _rhs: rustc_middle::mir::coverage::ExpressionOperandId,
        _region: Option<rustc_middle::mir::coverage::CodeRegion>,
    ) -> bool {
        todo!()
    }

    fn add_coverage_unreachable(
        &mut self,
        _instance: ty::Instance<'tcx>,
        _region: rustc_middle::mir::coverage::CodeRegion,
    ) -> bool {
        todo!()
    }
}

impl<'ll, 'tcx, 'a> BuilderMethods<'a, 'tcx> for Builder<'a, 'll, 'tcx> {
    fn build(cx: &'a CodegenCx<'ll, 'tcx>, llbb: &'ll BasicBlock) -> Self {
        let bx = Builder::with_cx(cx);
        unsafe {
            llvm::LLVMPositionBuilderAtEnd(&bx.llbuilder.lock().unwrap(), llbb);
        }
        bx
    }

    fn llbb(&self) -> &'ll BasicBlock {
        unsafe { llvm::LLVMGetInsertBlock(&mut self.llbuilder.lock().unwrap()) }
    }

    fn build_sibling_block(&mut self, name: &str) -> Self {
        let llbb = self.append_sibling_block(name);
        Self::build(self.cx, llbb)
    }

    fn set_span(&mut self, _span: Span) {}

    fn append_block(cx: &'a CodegenCx<'ll, 'tcx>, llfn: &'ll Value, name: &str) -> &'ll BasicBlock {
        unsafe {
            let name = CString::new(name).unwrap();
            llvm::LLVMAppendBasicBlockInContext(cx.llcx, llfn, name.as_ptr())
        }
    }

    fn append_sibling_block(&mut self, name: &str) -> &'ll BasicBlock {
        Self::append_block(self.cx, self.llfn(), name)
    }

    fn ret_void(&mut self) {
        trace!("Ret void");
        unsafe {
            llvm::LLVMBuildRetVoid(&mut self.llbuilder.lock().unwrap());
        }
    }

    fn ret(&mut self, mut v: &'ll Value) {
        trace!("Ret `{:?}`", v);
        unsafe {
            let ty = LLVMRustGetValueType(v);
            let (new_ty, changed) = get_transformed_type(self.cx, ty);
            if changed {
                v = crate::int_replace::transmute_llval(
                    *self.llbuilder.lock().unwrap(),
                    &self.cx,
                    v,
                    new_ty,
                );
            }
            llvm::LLVMBuildRet(&mut self.llbuilder.lock().unwrap(), v);
        }
    }

    fn br(&mut self, dest: &'ll BasicBlock) {
        trace!("Br");
        unsafe {
            llvm::LLVMBuildBr(&mut self.llbuilder.lock().unwrap(), dest);
        }
    }

    fn cond_br(
        &mut self,
        cond: &'ll Value,
        then_llbb: &'ll BasicBlock,
        else_llbb: &'ll BasicBlock,
    ) {
        trace!("Cond br `{:?}`", cond);
        unsafe {
            llvm::LLVMBuildCondBr(
                &mut self.llbuilder.lock().unwrap(),
                cond,
                then_llbb,
                else_llbb,
            );
        }
    }

    fn switch(
        &mut self,
        v: &'ll Value,
        else_llbb: &'ll BasicBlock,
        cases: impl ExactSizeIterator<Item = (u128, &'ll BasicBlock)>,
    ) {
        trace!("Switch `{:?}`", v);
        let switch = unsafe {
            llvm::LLVMBuildSwitch(
                &mut self.llbuilder.lock().unwrap(),
                v,
                else_llbb,
                cases.len() as c_uint,
            )
        };
        for (on_val, dest) in cases {
            let on_val = self.const_uint_big(self.val_ty(v), on_val);
            unsafe { llvm::LLVMAddCase(switch, on_val, dest) }
        }
    }

    fn invoke(
        &mut self,
        ty: &'ll Type,
        llfn: &'ll Value,
        args: &[&'ll Value],
        then: &'ll BasicBlock,
        _catch: &'ll BasicBlock,
        funclet: Option<&()>,
    ) -> &'ll Value {
        trace!("invoke");
        let call = self.call(ty, llfn, args, funclet);
        // exceptions arent a thing, go directly to the `then` block
        unsafe { llvm::LLVMBuildBr(&mut self.llbuilder.lock().unwrap(), then) };
        call
    }

    fn unreachable(&mut self) {
        trace!("Unreachable");
        unsafe {
            llvm::LLVMBuildUnreachable(&mut self.llbuilder.lock().unwrap());
        }
    }

    builder_methods_for_value_instructions! {
        add(a, b) => LLVMBuildAdd,
        fadd(a, b) => LLVMBuildFAdd,
        sub(a, b) => LLVMBuildSub,
        fsub(a, b) => LLVMBuildFSub,
        mul(a, b) => LLVMBuildMul,
        fmul(a, b) => LLVMBuildFMul,
        udiv(a, b) => LLVMBuildUDiv,
        exactudiv(a, b) => LLVMBuildExactUDiv,
        sdiv(a, b) => LLVMBuildSDiv,
        exactsdiv(a, b) => LLVMBuildExactSDiv,
        fdiv(a, b) => LLVMBuildFDiv,
        urem(a, b) => LLVMBuildURem,
        srem(a, b) => LLVMBuildSRem,
        frem(a, b) => LLVMBuildFRem,
        shl(a, b) => LLVMBuildShl,
        lshr(a, b) => LLVMBuildLShr,
        ashr(a, b) => LLVMBuildAShr,
        and(a, b) => LLVMBuildAnd,
        or(a, b) => LLVMBuildOr,
        xor(a, b) => LLVMBuildXor,
        neg(x) => LLVMBuildNeg,
        fneg(x) => LLVMBuildFNeg,
        not(x) => LLVMBuildNot,
        unchecked_sadd(x, y) => LLVMBuildNSWAdd,
        unchecked_uadd(x, y) => LLVMBuildNUWAdd,
        unchecked_ssub(x, y) => LLVMBuildNSWSub,
        unchecked_usub(x, y) => LLVMBuildNUWSub,
        unchecked_smul(x, y) => LLVMBuildNSWMul,
        unchecked_umul(x, y) => LLVMBuildNUWMul,
    }

    fn fadd_fast(&mut self, lhs: &'ll Value, rhs: &'ll Value) -> &'ll Value {
        unsafe {
            let instr =
                llvm::LLVMBuildFAdd(&mut self.llbuilder.lock().unwrap(), lhs, rhs, unnamed());
            llvm::LLVMRustSetFastMath(instr);
            instr
        }
    }

    fn fsub_fast(&mut self, lhs: &'ll Value, rhs: &'ll Value) -> &'ll Value {
        unsafe {
            let instr =
                llvm::LLVMBuildFSub(&mut self.llbuilder.lock().unwrap(), lhs, rhs, unnamed());
            llvm::LLVMRustSetFastMath(instr);
            instr
        }
    }

    fn fmul_fast(&mut self, lhs: &'ll Value, rhs: &'ll Value) -> &'ll Value {
        unsafe {
            let instr =
                llvm::LLVMBuildFMul(&mut self.llbuilder.lock().unwrap(), lhs, rhs, unnamed());
            llvm::LLVMRustSetFastMath(instr);
            instr
        }
    }

    fn fdiv_fast(&mut self, lhs: &'ll Value, rhs: &'ll Value) -> &'ll Value {
        unsafe {
            let instr =
                llvm::LLVMBuildFDiv(&mut self.llbuilder.lock().unwrap(), lhs, rhs, unnamed());
            llvm::LLVMRustSetFastMath(instr);
            instr
        }
    }

    fn frem_fast(&mut self, lhs: &'ll Value, rhs: &'ll Value) -> &'ll Value {
        unsafe {
            let instr =
                llvm::LLVMBuildFRem(&mut self.llbuilder.lock().unwrap(), lhs, rhs, unnamed());
            llvm::LLVMRustSetFastMath(instr);
            instr
        }
    }

    fn checked_binop(
        &mut self,
        oop: OverflowOp,
        ty: Ty<'_>,
        lhs: Self::Value,
        rhs: Self::Value,
    ) -> (Self::Value, Self::Value) {
        trace!(
            "Checked binop `{:?}`, lhs: `{:?}`, rhs: `{:?}`",
            ty,
            lhs,
            rhs
        );
        use rustc_middle::ty::IntTy::*;
        use rustc_middle::ty::UintTy::*;
        use rustc_middle::ty::{Int, Uint};

        let new_kind = match ty.kind() {
            Int(t @ Isize) => Int(t.normalize(self.tcx.sess.target.pointer_width)),
            Uint(t @ Usize) => Uint(t.normalize(self.tcx.sess.target.pointer_width)),
            t @ (Uint(_) | Int(_)) => t.clone(),
            _ => panic!("tried to get overflow intrinsic for op applied to non-int type"),
        };

        let name = match oop {
            OverflowOp::Add => match new_kind {
                Int(I8) => "__nvvm_i8_addo",
                Int(I16) => "llvm.sadd.with.overflow.i16",
                Int(I32) => "llvm.sadd.with.overflow.i32",
                Int(I64) => "llvm.sadd.with.overflow.i64",
                Int(I128) => "__nvvm_i128_addo",

                Uint(U8) => "__nvvm_u8_addo",
                Uint(U16) => "llvm.uadd.with.overflow.i16",
                Uint(U32) => "llvm.uadd.with.overflow.i32",
                Uint(U64) => "llvm.uadd.with.overflow.i64",
                Uint(U128) => "__nvvm_u128_addo",
                _ => unreachable!(),
            },
            OverflowOp::Sub => match new_kind {
                Int(I8) => "__nvvm_i8_subo",
                Int(I16) => "llvm.ssub.with.overflow.i16",
                Int(I32) => "llvm.ssub.with.overflow.i32",
                Int(I64) => "llvm.ssub.with.overflow.i64",
                Int(I128) => "__nvvm_i128_subo",

                Uint(U8) => "__nvvm_u8_subo",
                Uint(U16) => "llvm.usub.with.overflow.i16",
                Uint(U32) => "llvm.usub.with.overflow.i32",
                Uint(U64) => "llvm.usub.with.overflow.i64",
                Uint(U128) => "__nvvm_u128_subo",

                _ => unreachable!(),
            },
            OverflowOp::Mul => match new_kind {
                Int(I8) => "__nvvm_i8_mulo",
                Int(I16) => "llvm.smul.with.overflow.i16",
                Int(I32) => "llvm.smul.with.overflow.i32",
                Int(I64) => "llvm.smul.with.overflow.i64",
                Int(I128) => "__nvvm_i128_mulo",

                Uint(U8) => "__nvvm_u8_mulo",
                Uint(U16) => "llvm.umul.with.overflow.i16",
                Uint(U32) => "llvm.umul.with.overflow.i32",
                Uint(U64) => "llvm.umul.with.overflow.i64",
                Uint(U128) => "__nvvm_u128_mulo",

                _ => unreachable!(),
            },
        };

        let intrinsic = self.get_intrinsic(&name);
        // call actually ignores the ty param for now, we just need it for conformance with nightly api
        // so give it a dummy type
        let res = self.call(self.type_i1(), intrinsic, &[lhs, rhs], None);
        (self.extract_value(res, 0), self.extract_value(res, 1))
    }

    fn from_immediate(&mut self, val: Self::Value) -> Self::Value {
        if self.cx().val_ty(val) == self.cx().type_i1() {
            self.zext(val, self.cx().type_i8())
        } else {
            val
        }
    }
    fn to_immediate_scalar(&mut self, val: Self::Value, scalar: abi::Scalar) -> Self::Value {
        if scalar.is_bool() {
            return self.trunc(val, self.cx().type_i1());
        }
        val
    }

    fn alloca(&mut self, ty: &'ll Type, align: Align) -> &'ll Value {
        trace!("Alloca `{:?}`", ty);
        let mut bx = Builder::with_cx(self.cx);
        bx.position_at_start(unsafe { llvm::LLVMGetFirstBasicBlock(self.llfn()) });
        bx.dynamic_alloca(ty, align)
    }

    fn dynamic_alloca(&mut self, ty: &'ll Type, align: Align) -> &'ll Value {
        trace!("Dynamic Alloca `{:?}`", ty);
        unsafe {
            let alloca = llvm::LLVMBuildAlloca(&mut self.llbuilder.lock().unwrap(), ty, unnamed());
            llvm::LLVMSetAlignment(alloca, align.bytes() as c_uint);
            alloca
        }
    }

    fn array_alloca(&mut self, ty: &'ll Type, len: &'ll Value, align: Align) -> &'ll Value {
        unsafe {
            let alloca =
                llvm::LLVMBuildArrayAlloca(&mut self.llbuilder.lock().unwrap(), ty, len, unnamed());
            llvm::LLVMSetAlignment(alloca, align.bytes() as c_uint);
            alloca
        }
    }

    fn load(&mut self, _: &'ll Type, ptr: &'ll Value, align: Align) -> &'ll Value {
        trace!("Load `{:?}`", ptr);
        unsafe {
            let load = llvm::LLVMBuildLoad(&mut self.llbuilder.lock().unwrap(), ptr, unnamed());
            llvm::LLVMSetAlignment(load, align.bytes() as c_uint);
            load
        }
    }

    fn volatile_load(&mut self, _: &'ll Type, ptr: &'ll Value) -> &'ll Value {
        trace!("Volatile load `{:?}`", ptr);
        unsafe {
            let load = llvm::LLVMBuildLoad(&mut self.llbuilder.lock().unwrap(), ptr, unnamed());
            llvm::LLVMSetVolatile(load, llvm::True);
            load
        }
    }

    fn atomic_load(
        &mut self,
        ty: &'ll Type,
        ptr: &'ll Value,
        _order: AtomicOrdering,
        _size: Size,
    ) -> &'ll Value {
        // core seems to think that nvptx has atomic loads, which is not true for NVVM IR,
        // therefore our only option is to print that this is not supported then trap.
        // i have heard of cursed things such as emulating this with __threadfence and volatile loads
        // but that needs to be experimented with in terms of safety and behavior.
        // NVVM has explicit intrinsics for adding and subtracting floats which we expose elsewhere

        // TODO(RDambrosio016): is there a way we can just generate a panic with a message instead
        // of doing this ourselves? since all panics will be aborts, it should be equivalent
        // let message = "Atomic Loads are not supported in CUDA.\0";

        // let vprintf = self.get_intrinsic("vprintf");
        // let formatlist = self.const_str(Symbol::intern(message)).0;
        // let valist = self.const_null(self.type_void());

        // self.call(vprintf, &[formatlist, valist], None);

        let trap = self.get_intrinsic("llvm.trap");
        self.call(ty, trap, &[], None);
        unsafe {
            let load = llvm::LLVMBuildLoad(&mut self.llbuilder.lock().unwrap(), ptr, unnamed());
            load
        }
    }

    fn load_operand(&mut self, place: PlaceRef<'tcx, &'ll Value>) -> OperandRef<'tcx, &'ll Value> {
        trace!("Load operand `{:?}`", place);
        assert_eq!(place.llextra.is_some(), place.layout.is_unsized());

        if place.layout.is_zst() {
            return OperandRef::new_zst(self, place.layout);
        }

        fn scalar_load_metadata<'a, 'll, 'tcx>(
            bx: &mut Builder<'a, 'll, 'tcx>,
            load: &'ll Value,
            scalar: &abi::Scalar,
        ) {
            match scalar.value {
                abi::Int(..) => {
                    if !scalar.is_always_valid(bx) {
                        bx.range_metadata(load, scalar.valid_range);
                    }
                }
                abi::Pointer if !scalar.valid_range.contains(0) => {
                    bx.nonnull_metadata(load);
                }
                _ => {}
            }
        }

        let val = if let Some(llextra) = place.llextra {
            OperandValue::Ref(place.llval, Some(llextra), place.align)
        } else if place.layout.is_llvm_immediate() {
            let mut const_llval = None;
            unsafe {
                if let Some(global) = llvm::LLVMIsAGlobalVariable(place.llval) {
                    if llvm::LLVMIsGlobalConstant(global) == llvm::True {
                        const_llval = llvm::LLVMGetInitializer(global);
                    }
                }
            }

            let llval = const_llval.unwrap_or_else(|| {
                let load = self.load(place.layout.llvm_type(self), place.llval, place.align);

                if let abi::Abi::Scalar(ref scalar) = place.layout.abi {
                    scalar_load_metadata(self, load, scalar);
                }
                load
            });

            OperandValue::Immediate(self.to_immediate(llval, place.layout))
        } else if let abi::Abi::ScalarPair(ref a, ref b) = place.layout.abi {
            let b_offset = a.value.size(self).align_to(b.value.align(self).abi);
            let pair_ty = place.layout.llvm_type(self);

            let mut load = |i, scalar: &abi::Scalar, align| {
                let llptr = self.struct_gep(pair_ty, place.llval, i as u64);
                let llty = place.layout.scalar_pair_element_llvm_type(self, i, false);
                let load = self.load(llty, llptr, align);
                scalar_load_metadata(self, load, scalar);
                self.to_immediate_scalar(load, *scalar)
            };

            let val = OperandValue::Pair(
                load(0, a, place.align),
                load(1, b, place.align.restrict_for_offset(b_offset)),
            );

            val
        } else {
            OperandValue::Ref(place.llval, None, place.align)
        };

        OperandRef {
            val,
            layout: place.layout,
        }
    }

    fn write_operand_repeatedly(
        mut self,
        cg_elem: OperandRef<'tcx, &'ll Value>,
        count: u64,
        dest: PlaceRef<'tcx, &'ll Value>,
    ) -> Self {
        trace!("write operand repeatedly");
        let zero = self.const_usize(0);
        let count = self.const_usize(count);
        let start = dest.project_index(&mut self, zero).llval;
        let end = dest.project_index(&mut self, count).llval;

        let mut header_bx = self.build_sibling_block("repeat_loop_header");
        let mut body_bx = self.build_sibling_block("repeat_loop_body");
        let next_bx = self.build_sibling_block("repeat_loop_next");

        self.br(header_bx.llbb());
        let current = header_bx.phi(self.val_ty(start), &[start], &[self.llbb()]);

        let keep_going = header_bx.icmp(IntPredicate::IntNE, current, end);
        header_bx.cond_br(keep_going, body_bx.llbb(), next_bx.llbb());

        let align = dest
            .align
            .restrict_for_offset(dest.layout.field(self.cx(), 0).size);
        cg_elem.val.store(
            &mut body_bx,
            PlaceRef::new_sized_aligned(current, cg_elem.layout, align),
        );

        let next = body_bx.inbounds_gep(
            self.backend_type(cg_elem.layout),
            current,
            &[self.const_usize(1)],
        );
        body_bx.br(header_bx.llbb());
        header_bx.add_incoming_to_phi(current, next, body_bx.llbb());

        next_bx
    }

    fn range_metadata(&mut self, load: &'ll Value, range: WrappingRange) {
        trace!("range metadata on {:?}: {:?}", load, range);
        unsafe {
            let llty = self.cx.val_ty(load);
            let v = [
                self.cx.const_uint_big(llty, range.start),
                self.cx.const_uint_big(llty, range.end.wrapping_add(1)),
            ];

            llvm::LLVMSetMetadata(
                load,
                llvm::MetadataType::MD_range as c_uint,
                llvm::LLVMMDNodeInContext(self.cx.llcx, v.as_ptr(), v.len() as c_uint),
            );
        }
    }

    fn nonnull_metadata(&mut self, load: &'ll Value) {
        unsafe {
            llvm::LLVMSetMetadata(
                load,
                llvm::MetadataType::MD_nonnull as c_uint,
                llvm::LLVMMDNodeInContext(self.cx.llcx, ptr::null(), 0),
            );
        }
    }

    fn store(&mut self, val: &'ll Value, ptr: &'ll Value, align: Align) -> &'ll Value {
        trace!("Store val `{:?}` into ptr `{:?}`", val, ptr);
        self.store_with_flags(val, ptr, align, MemFlags::empty())
    }

    fn store_with_flags(
        &mut self,
        val: &'ll Value,
        ptr: &'ll Value,
        align: Align,
        flags: MemFlags,
    ) -> &'ll Value {
        let ptr = self.check_store(val, ptr);
        unsafe {
            let store = llvm::LLVMBuildStore(&mut self.llbuilder.lock().unwrap(), val, ptr);
            let align = if flags.contains(MemFlags::UNALIGNED) {
                1
            } else {
                align.bytes() as c_uint
            };
            llvm::LLVMSetAlignment(store, align);
            if flags.contains(MemFlags::VOLATILE) {
                llvm::LLVMSetVolatile(store, llvm::True);
            }
            if flags.contains(MemFlags::NONTEMPORAL) {
                // According to LLVM [1] building a nontemporal store must
                // *always* point to a metadata value of the integer 1.
                //
                // [1]: http://llvm.org/docs/LangRef.html#store-instruction
                let one = self.cx.const_i32(1);
                let node = llvm::LLVMMDNodeInContext(self.cx.llcx, &one, 1);
                llvm::LLVMSetMetadata(store, llvm::MetadataType::MD_nontemporal as c_uint, node);
            }

            store
        }
    }

    fn atomic_store(
        &mut self,
        _val: &'ll Value,
        ptr: &'ll Value,
        _order: rustc_codegen_ssa::common::AtomicOrdering,
        _size: Size,
    ) {
        // see comment in atomic_load

        // let message = "Atomic Stores are not supported in CUDA.\0";

        // let vprintf = self.get_intrinsic("vprintf");
        // let formatlist = self.const_str(Symbol::intern(message)).0;
        // let valist = self.const_null(self.type_void());

        // self.call(vprintf, &[formatlist, valist], None);
        self.abort();
        unsafe {
            llvm::LLVMBuildLoad(&mut self.llbuilder.lock().unwrap(), ptr, unnamed());
        }
    }

    fn gep(&mut self, _: &'ll Type, ptr: &'ll Value, indices: &[&'ll Value]) -> &'ll Value {
        trace!("gep: {:?} with indices {:?}", ptr, indices);
        unsafe {
            llvm::LLVMBuildGEP(
                &mut self.llbuilder.lock().unwrap(),
                ptr,
                indices.as_ptr(),
                indices.len() as c_uint,
                unnamed(),
            )
        }
    }

    fn inbounds_gep(
        &mut self,
        _: &'ll Type,
        ptr: &'ll Value,
        indices: &[&'ll Value],
    ) -> &'ll Value {
        trace!("gep inbounds: {:?} with indices {:?}", ptr, indices);
        unsafe {
            llvm::LLVMBuildInBoundsGEP(
                &mut self.llbuilder.lock().unwrap(),
                ptr,
                indices.as_ptr(),
                indices.len() as c_uint,
                unnamed(),
            )
        }
    }

    fn struct_gep(&mut self, _: &'ll Type, ptr: &'ll Value, idx: u64) -> &'ll Value {
        trace!("struct gep: {:?}, {:?}", ptr, idx);
        assert_eq!(idx as c_uint as u64, idx);
        unsafe {
            llvm::LLVMBuildStructGEP(
                &mut self.llbuilder.lock().unwrap(),
                ptr,
                idx as c_uint,
                unnamed(),
            )
        }
    }

    /* Casts */
    fn trunc(&mut self, val: &'ll Value, dest_ty: &'ll Type) -> &'ll Value {
        trace!("trunc {:?} to {:?}", val, dest_ty);
        unsafe {
            llvm::LLVMBuildTrunc(&mut self.llbuilder.lock().unwrap(), val, dest_ty, unnamed())
        }
    }

    fn sext(&mut self, val: &'ll Value, dest_ty: &'ll Type) -> &'ll Value {
        trace!("sext {:?} to {:?}", val, dest_ty);
        unsafe { llvm::LLVMBuildSExt(&mut self.llbuilder.lock().unwrap(), val, dest_ty, unnamed()) }
    }

    fn fptoui_sat(&mut self, _val: &'ll Value, _dest_ty: &'ll Type) -> Option<&'ll Value> {
        None
    }

    fn fptosi_sat(&mut self, _val: &'ll Value, _dest_ty: &'ll Type) -> Option<&'ll Value> {
        None
    }

    // fn fptosui_may_trap(&self, val: &'ll Value, dest_ty: &'ll Type) -> bool {
    //     false
    // }

    fn fptoui(&mut self, val: &'ll Value, dest_ty: &'ll Type) -> &'ll Value {
        trace!("fptoui {:?} to {:?}", val, dest_ty);
        unsafe {
            llvm::LLVMBuildFPToUI(&mut self.llbuilder.lock().unwrap(), val, dest_ty, unnamed())
        }
    }

    fn fptosi(&mut self, val: &'ll Value, dest_ty: &'ll Type) -> &'ll Value {
        trace!("fptosi {:?} to {:?}", val, dest_ty);
        unsafe {
            llvm::LLVMBuildFPToSI(&mut self.llbuilder.lock().unwrap(), val, dest_ty, unnamed())
        }
    }

    fn uitofp(&mut self, val: &'ll Value, dest_ty: &'ll Type) -> &'ll Value {
        trace!("uitofp {:?} to {:?}", val, dest_ty);
        unsafe {
            llvm::LLVMBuildUIToFP(&mut self.llbuilder.lock().unwrap(), val, dest_ty, unnamed())
        }
    }

    fn sitofp(&mut self, val: &'ll Value, dest_ty: &'ll Type) -> &'ll Value {
        trace!("sitofp {:?} to {:?}", val, dest_ty);
        unsafe {
            llvm::LLVMBuildSIToFP(&mut self.llbuilder.lock().unwrap(), val, dest_ty, unnamed())
        }
    }

    fn fptrunc(&mut self, val: &'ll Value, dest_ty: &'ll Type) -> &'ll Value {
        trace!("fptrunc {:?} to {:?}", val, dest_ty);
        unsafe {
            llvm::LLVMBuildFPTrunc(&mut self.llbuilder.lock().unwrap(), val, dest_ty, unnamed())
        }
    }

    fn fpext(&mut self, val: &'ll Value, dest_ty: &'ll Type) -> &'ll Value {
        trace!("fpext {:?} to {:?}", val, dest_ty);
        unsafe {
            llvm::LLVMBuildFPExt(&mut self.llbuilder.lock().unwrap(), val, dest_ty, unnamed())
        }
    }

    fn ptrtoint(&mut self, val: &'ll Value, dest_ty: &'ll Type) -> &'ll Value {
        trace!("ptrtoint {:?} to {:?}", val, dest_ty);
        unsafe {
            llvm::LLVMBuildPtrToInt(&mut self.llbuilder.lock().unwrap(), val, dest_ty, unnamed())
        }
    }

    fn inttoptr(&mut self, val: &'ll Value, dest_ty: &'ll Type) -> &'ll Value {
        trace!("inttoptr {:?} to {:?}", val, dest_ty);
        unsafe {
            llvm::LLVMBuildIntToPtr(&mut self.llbuilder.lock().unwrap(), val, dest_ty, unnamed())
        }
    }

    fn bitcast(&mut self, val: &'ll Value, dest_ty: &'ll Type) -> &'ll Value {
        trace!("Bitcast `{:?}` to ty `{:?}`", val, dest_ty);
        unsafe {
            llvm::LLVMBuildBitCast(&mut self.llbuilder.lock().unwrap(), val, dest_ty, unnamed())
        }
    }

    fn intcast(&mut self, val: &'ll Value, dest_ty: &'ll Type, is_signed: bool) -> &'ll Value {
        trace!("Intcast `{:?}` to ty `{:?}`", val, dest_ty);
        unsafe {
            llvm::LLVMRustBuildIntCast(&mut self.llbuilder.lock().unwrap(), val, dest_ty, is_signed)
        }
    }

    fn pointercast(&mut self, val: &'ll Value, dest_ty: &'ll Type) -> &'ll Value {
        trace!("Pointercast `{:?}` to ty `{:?}`", val, dest_ty);
        unsafe {
            llvm::LLVMBuildPointerCast(&mut self.llbuilder.lock().unwrap(), val, dest_ty, unnamed())
        }
    }

    /* Comparisons */
    fn icmp(&mut self, op: IntPredicate, lhs: &'ll Value, rhs: &'ll Value) -> &'ll Value {
        trace!("Icmp lhs: `{:?}`, rhs: `{:?}`", lhs, rhs);
        unsafe {
            let op = llvm::IntPredicate::from_generic(op);
            llvm::LLVMBuildICmp(
                &mut self.llbuilder.lock().unwrap(),
                op as c_uint,
                lhs,
                rhs,
                unnamed(),
            )
        }
    }

    fn fcmp(&mut self, op: RealPredicate, lhs: &'ll Value, rhs: &'ll Value) -> &'ll Value {
        trace!("Fcmp lhs: `{:?}`, rhs: `{:?}`", lhs, rhs);
        unsafe {
            llvm::LLVMBuildFCmp(
                &mut self.llbuilder.lock().unwrap(),
                op as c_uint,
                lhs,
                rhs,
                unnamed(),
            )
        }
    }

    /* Miscellaneous instructions */
    fn memcpy(
        &mut self,
        dst: &'ll Value,
        dst_align: Align,
        src: &'ll Value,
        src_align: Align,
        size: &'ll Value,
        flags: MemFlags,
    ) {
        assert!(
            !flags.contains(MemFlags::NONTEMPORAL),
            "non-temporal memcpy not supported"
        );
        let size = self.intcast(size, self.type_isize(), false);
        let is_volatile = flags.contains(MemFlags::VOLATILE);
        let dst = self.pointercast(dst, self.type_i8p());
        let src = self.pointercast(src, self.type_i8p());
        unsafe {
            llvm::LLVMRustBuildMemCpy(
                &mut self.llbuilder.lock().unwrap(),
                dst,
                dst_align.bytes() as c_uint,
                src,
                src_align.bytes() as c_uint,
                size,
                is_volatile,
            );
        }
    }

    fn memmove(
        &mut self,
        dst: &'ll Value,
        dst_align: Align,
        src: &'ll Value,
        src_align: Align,
        size: &'ll Value,
        flags: MemFlags,
    ) {
        assert!(
            !flags.contains(MemFlags::NONTEMPORAL),
            "non-temporal memmove not supported"
        );
        let size = self.intcast(size, self.type_isize(), false);
        let is_volatile = flags.contains(MemFlags::VOLATILE);
        let dst = self.pointercast(dst, self.type_i8p());
        let src = self.pointercast(src, self.type_i8p());
        unsafe {
            llvm::LLVMRustBuildMemMove(
                &mut self.llbuilder.lock().unwrap(),
                dst,
                dst_align.bytes() as c_uint,
                src,
                src_align.bytes() as c_uint,
                size,
                is_volatile,
            );
        }
    }

    fn memset(
        &mut self,
        ptr: &'ll Value,
        fill_byte: &'ll Value,
        size: &'ll Value,
        align: Align,
        flags: MemFlags,
    ) {
        let is_volatile = flags.contains(MemFlags::VOLATILE);
        let ptr = self.pointercast(ptr, self.type_i8p());
        unsafe {
            llvm::LLVMRustBuildMemSet(
                &mut self.llbuilder.lock().unwrap(),
                ptr,
                align.bytes() as c_uint,
                fill_byte,
                size,
                is_volatile,
            );
        }
    }

    fn select(
        &mut self,
        mut cond: &'ll Value,
        then_val: &'ll Value,
        else_val: &'ll Value,
    ) -> &'ll Value {
        unsafe {
            if llvm::LLVMRustGetValueType(cond) == llvm::LLVMVectorType(self.type_i1(), 2) {
                cond = self.const_bool(false);
            }
            llvm::LLVMBuildSelect(
                &mut self.llbuilder.lock().unwrap(),
                cond,
                then_val,
                else_val,
                unnamed(),
            )
        }
    }

    fn va_arg(&mut self, list: &'ll Value, ty: &'ll Type) -> &'ll Value {
        unsafe { llvm::LLVMBuildVAArg(&mut self.llbuilder.lock().unwrap(), list, ty, unnamed()) }
    }

    fn extract_element(&mut self, vec: &'ll Value, idx: &'ll Value) -> &'ll Value {
        trace!("extract element {:?}, {:?}", vec, idx);
        unsafe {
            llvm::LLVMBuildExtractElement(&mut self.llbuilder.lock().unwrap(), vec, idx, unnamed())
        }
    }

    fn vector_splat(&mut self, _num_elts: usize, _elt: &'ll Value) -> &'ll Value {
        self.unsupported("vector splats");
    }

    fn extract_value(&mut self, agg_val: &'ll Value, idx: u64) -> &'ll Value {
        trace!("extract value {:?}, {:?}", agg_val, idx);
        assert_eq!(idx as c_uint as u64, idx);
        unsafe {
            llvm::LLVMBuildExtractValue(
                &mut self.llbuilder.lock().unwrap(),
                agg_val,
                idx as c_uint,
                unnamed(),
            )
        }
    }

    fn insert_value(&mut self, agg_val: &'ll Value, elt: &'ll Value, idx: u64) -> &'ll Value {
        trace!("insert value {:?}, {:?}", agg_val, idx);
        assert_eq!(idx as c_uint as u64, idx);
        unsafe {
            llvm::LLVMBuildInsertValue(
                &mut self.llbuilder.lock().unwrap(),
                agg_val,
                elt,
                idx as c_uint,
                unnamed(),
            )
        }
    }

    fn landing_pad(
        &mut self,
        _ty: &'ll Type,
        _pers_fn: &'ll Value,
        _num_clauses: usize,
    ) -> &'ll Value {
        todo!();
    }

    fn set_cleanup(&mut self, _landing_pad: &'ll Value) {}

    fn resume(&mut self, _exn: &'ll Value) -> &'ll Value {
        self.unsupported("resumes");
    }

    fn cleanup_pad(&mut self, _parent: Option<&'ll Value>, _args: &[&'ll Value]) -> () {
        ()
    }

    fn cleanup_ret(&mut self, _funclet: &(), _unwind: Option<&'ll BasicBlock>) -> &'ll Value {
        // rustc doesnt actually use this value ;)
        self.const_bool(false)
    }

    fn catch_pad(&mut self, _parent: &'ll Value, _args: &[&'ll Value]) -> () {
        ()
    }

    fn catch_switch(
        &mut self,
        _parent: Option<&'ll Value>,
        _unwind: Option<&'ll BasicBlock>,
        _num_handlers: usize,
    ) -> &'ll Value {
        self.unsupported("catch switches");
    }

    fn add_handler(&mut self, _catch_switch: &'ll Value, _handler: &'ll BasicBlock) {
        todo!();
    }

    fn set_personality_fn(&mut self, _personality: &'ll Value) {}

    // Atomic Operations
    fn atomic_cmpxchg(
        &mut self,
        _dst: &'ll Value,
        _cmp: &'ll Value,
        _src: &'ll Value,
        _order: rustc_codegen_ssa::common::AtomicOrdering,
        _failure_order: rustc_codegen_ssa::common::AtomicOrdering,
        _weak: bool,
    ) -> &'ll Value {
        // allowed but only for some things and with restrictions
        // https://docs.nvidia.com/cuda/nvvm-ir-spec/index.html#cmpxchg-instruction
        self.fatal("atomic cmpxchg is not supported")
    }
    fn atomic_rmw(
        &mut self,
        _op: rustc_codegen_ssa::common::AtomicRmwBinOp,
        _dst: &'ll Value,
        _src: &'ll Value,
        _order: rustc_codegen_ssa::common::AtomicOrdering,
    ) -> &'ll Value {
        // see cmpxchg comment
        self.fatal("atomic rmw is not supported")
    }

    fn atomic_fence(
        &mut self,
        _order: rustc_codegen_ssa::common::AtomicOrdering,
        _scope: rustc_codegen_ssa::common::SynchronizationScope,
    ) {
        self.fatal("atomic fence is not supported, use cuda_std intrinsics instead")
    }

    fn set_invariant_load(&mut self, load: &'ll Value) {
        unsafe {
            llvm::LLVMSetMetadata(
                load,
                llvm::MetadataType::MD_invariant_load as c_uint,
                llvm::LLVMMDNodeInContext(self.cx.llcx, ptr::null(), 0),
            );
        }
    }

    fn lifetime_start(&mut self, ptr: &'ll Value, size: Size) {
        self.call_lifetime_intrinsic("llvm.lifetime.start.p0i8", ptr, size);
    }

    fn lifetime_end(&mut self, ptr: &'ll Value, size: Size) {
        self.call_lifetime_intrinsic("llvm.lifetime.end.p0i8", ptr, size);
    }

    fn instrprof_increment(
        &mut self,
        _fn_name: &'ll Value,
        _hash: &'ll Value,
        _num_counters: &'ll Value,
        _index: &'ll Value,
    ) {
        todo!();
    }

    fn call(
        &mut self,
        _: &'ll Type,
        llfn: &'ll Value,
        args: &[&'ll Value],
        _funclet: Option<&()>,
    ) -> &'ll Value {
        trace!("Calling fn {:?} with args {:?}", llfn, args);
        self.cx.last_call_llfn.set(None);
        let args = self.check_call("call", llfn, args);

        let mut ret = unsafe {
            llvm::LLVMRustBuildCall(
                &mut self.llbuilder.lock().unwrap(),
                llfn,
                args.as_ptr() as *const &llvm::Value,
                args.len() as c_uint,
                None,
            )
        };

        // bitcast return type if the type was remapped
        let map = self.cx.remapped_integer_args.borrow();
        let mut fn_ty = unsafe { LLVMRustGetValueType(llfn) };
        while self.cx.type_kind(fn_ty) == TypeKind::Pointer {
            fn_ty = self.cx.element_type(fn_ty);
        }
        if let Some((Some(ret_ty), _)) = map.get(fn_ty) {
            self.cx.last_call_llfn.set(Some(ret));
            ret = transmute_llval(&mut *self.llbuilder.lock().unwrap(), &self.cx, ret, ret_ty);
        }

        ret
    }

    fn zext(&mut self, val: &'ll Value, dest_ty: &'ll Type) -> &'ll Value {
        trace!("Zext {:?} to {:?}", val, dest_ty);
        unsafe { llvm::LLVMBuildZExt(&mut self.llbuilder.lock().unwrap(), val, dest_ty, unnamed()) }
    }

    fn cx(&self) -> &CodegenCx<'ll, 'tcx> {
        self.cx
    }

    fn do_not_inline(&mut self, llret: &'ll Value) {
        llvm::Attribute::NoInline.apply_callsite(llvm::AttributePlace::Function, llret);
    }
}

impl<'a, 'll, 'tcx> StaticBuilderMethods for Builder<'a, 'll, 'tcx> {
    fn get_static(&mut self, def_id: DefId) -> &'ll Value {
        // Forward to the `get_static` method of `CodegenCx`
        self.cx().get_static(def_id)
    }
}

impl<'a, 'll, 'tcx> Builder<'a, 'll, 'tcx> {
    // TODO(RDambrosio016): fix this when nvidia fixes i128
    pub(crate) fn abort_and_ret_i128(&mut self) -> &'ll Value {
        self.abort();
        let first = self.const_u64(0);
        let second = self.const_u64(0);
        let vals = [first, second];
        unsafe { llvm::LLVMConstVector(vals.as_ptr(), 2) }
    }

    fn with_cx(cx: &'a CodegenCx<'ll, 'tcx>) -> Self {
        // Create a fresh builder from the crate context.
        let llbuilder = Mutex::new(unsafe { llvm::LLVMCreateBuilderInContext(cx.llcx) });
        Builder { llbuilder, cx }
    }

    pub fn llfn(&self) -> &'ll Value {
        unsafe { llvm::LLVMGetBasicBlockParent(self.llbb()) }
    }

    fn position_at_start(&mut self, llbb: &'ll BasicBlock) {
        unsafe {
            llvm::LLVMRustPositionBuilderAtStart(&mut self.llbuilder.lock().unwrap(), llbb);
        }
    }

    fn check_store(&mut self, val: &'ll Value, ptr: &'ll Value) -> &'ll Value {
        let dest_ptr_ty = self.cx.val_ty(ptr);
        let stored_ty = self.cx.val_ty(val);
        let stored_ptr_ty = self.cx.type_ptr_to(stored_ty);

        assert_eq!(self.cx.type_kind(dest_ptr_ty), TypeKind::Pointer);

        if dest_ptr_ty == stored_ptr_ty {
            ptr
        } else {
            self.bitcast(ptr, stored_ptr_ty)
        }
    }

    fn check_call<'b>(
        &mut self,
        typ: &str,
        llfn: &'ll Value,
        args: &'b [&'ll Value],
    ) -> Cow<'b, [&'ll Value]> {
        let mut fn_ty = self.cx.val_ty(llfn);
        // Strip off pointers
        while self.cx.type_kind(fn_ty) == TypeKind::Pointer {
            fn_ty = self.cx.element_type(fn_ty);
        }

        assert!(
            self.cx.type_kind(fn_ty) == TypeKind::Function,
            "builder::{} not passed a function, but {:?}",
            typ,
            fn_ty
        );

        let param_tys = self.cx.func_params_types(fn_ty);

        let all_args_match = param_tys
            .iter()
            .zip(args.iter().map(|&v| self.val_ty(v)))
            .all(|(expected_ty, actual_ty)| *expected_ty == actual_ty);

        if all_args_match {
            return Cow::Borrowed(args);
        }

        let casted_args: Vec<_> = param_tys
            .into_iter()
            .zip(args.iter())
            .enumerate()
            .map(|(_, (expected_ty, &actual_val))| {
                let actual_ty = self.val_ty(actual_val);

                if expected_ty != actual_ty {
                    trace!(
                        "Expected arg to be {:?} but instead found {:?}, bitcasting the pain away",
                        expected_ty,
                        actual_ty
                    );
                    self.bitcast(actual_val, expected_ty)
                } else {
                    actual_val
                }
            })
            .collect();

        Cow::Owned(casted_args)
    }

    pub fn va_arg(&mut self, list: &'ll Value, ty: &'ll Type) -> &'ll Value {
        unsafe { llvm::LLVMBuildVAArg(&mut self.llbuilder.lock().unwrap(), list, ty, unnamed()) }
    }

    fn call_lifetime_intrinsic(&mut self, intrinsic: &'static str, ptr: &'ll Value, size: Size) {
        let size = size.bytes();
        if size == 0 {
            return;
        }

        if !self.cx().sess().emit_lifetime_markers() {
            return;
        }

        let lifetime_intrinsic = self.cx.get_intrinsic(intrinsic);

        let ptr = self.pointercast(ptr, self.cx.type_i8p());
        self.call(
            self.type_i1(),
            lifetime_intrinsic,
            &[self.cx.const_u64(size), ptr],
            None,
        );
    }

    pub(crate) fn phi(
        &mut self,
        ty: &'ll Type,
        vals: &[&'ll Value],
        bbs: &[&'ll BasicBlock],
    ) -> &'ll Value {
        assert_eq!(vals.len(), bbs.len());
        let phi = unsafe { llvm::LLVMBuildPhi(&mut self.llbuilder.lock().unwrap(), ty, unnamed()) };
        unsafe {
            llvm::LLVMAddIncoming(phi, vals.as_ptr(), bbs.as_ptr(), vals.len() as c_uint);
            phi
        }
    }

    fn add_incoming_to_phi(&mut self, phi: &'ll Value, val: &'ll Value, bb: &'ll BasicBlock) {
        unsafe {
            llvm::LLVMAddIncoming(phi, &val, &bb, 1 as c_uint);
        }
    }
}
