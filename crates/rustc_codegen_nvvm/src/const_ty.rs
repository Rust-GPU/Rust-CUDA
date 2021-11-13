use crate::llvm::{self, Bool, False, True, Type, Value};
use crate::{consts::const_alloc_to_llvm, context::CodegenCx, target, ty::LayoutLlvmExt};
use abi::Primitive::Pointer;
use libc::c_uint;
use rustc_ast::Mutability;
use rustc_codegen_ssa::{
    mir::place::PlaceRef,
    traits::{BaseTypeMethods, ConstMethods, DerivedTypeMethods, MiscMethods, StaticMethods},
};
use rustc_middle::mir::interpret::{Allocation, GlobalAlloc, Scalar};
use rustc_middle::ty::layout::LayoutOf;
use rustc_middle::ty::{layout::TyAndLayout, ScalarInt};
use rustc_span::Symbol;
use rustc_target::abi::{self, AddressSpace, HasDataLayout, Size};
use tracing::trace;

impl<'ll, 'tcx> ConstMethods<'tcx> for CodegenCx<'ll, 'tcx> {
    fn const_data_from_alloc(&self, alloc: &Allocation) -> &'ll Value {
        const_alloc_to_llvm(self, alloc)
    }

    fn const_null(&self, t: &'ll Type) -> &'ll Value {
        unsafe { llvm::LLVMConstNull(t) }
    }

    fn const_undef(&self, t: &'ll Type) -> &'ll Value {
        unsafe { llvm::LLVMGetUndef(t) }
    }

    fn const_int(&self, t: &'ll Type, i: i64) -> &'ll Value {
        unsafe { llvm::LLVMConstInt(t, i as u64, True) }
    }

    fn const_uint(&self, t: &'ll Type, i: u64) -> &'ll Value {
        unsafe { llvm::LLVMConstInt(t, i, False) }
    }

    fn const_uint_big(&self, t: &'ll Type, u: u128) -> &'ll Value {
        unsafe {
            let words = [u as u64, (u >> 64) as u64];
            llvm::LLVMConstIntOfArbitraryPrecision(t, 2, words.as_ptr())
        }
    }

    fn const_bool(&self, val: bool) -> &'ll Value {
        self.const_uint(self.type_i1(), val as u64)
    }

    fn const_i32(&self, i: i32) -> &'ll Value {
        self.const_int(self.type_i32(), i as i64)
    }

    fn const_u32(&self, i: u32) -> &'ll Value {
        self.const_uint(self.type_i32(), i as u64)
    }

    fn const_u64(&self, i: u64) -> &'ll Value {
        self.const_uint(self.type_i64(), i)
    }

    fn const_usize(&self, i: u64) -> &'ll Value {
        let bit_size = target::pointer_size();
        if bit_size == 32 {
            // shouldnt happen but make sure it doesnt overflow
            // and the entire codegen burns down to the ground
            assert!(i < (1 << bit_size));
        }

        self.const_uint(self.isize_ty, i)
    }

    fn const_u8(&self, i: u8) -> &'ll Value {
        self.const_uint(self.type_i8(), i as u64)
    }

    fn const_real(&self, t: &'ll Type, val: f64) -> &'ll Value {
        unsafe { llvm::LLVMConstReal(t, val) }
    }

    fn const_str(&self, s: Symbol) -> (&'ll Value, &'ll Value) {
        let len = s.as_str().len();
        let val = self.const_cstr(s, false);
        let ty = self.type_ptr_to(self.layout_of(self.tcx.types.str_).llvm_type(self));
        let cs = unsafe { llvm::LLVMConstPointerCast(val, ty) };
        (cs, self.const_usize(len as u64))
    }

    fn const_struct(&self, elts: &[&'ll Value], packed: bool) -> &'ll Value {
        unsafe {
            llvm::LLVMConstStructInContext(
                self.llcx,
                elts.as_ptr(),
                elts.len() as c_uint,
                packed as Bool,
            )
        }
    }

    fn const_to_opt_uint(&self, v: &'ll Value) -> Option<u64> {
        unsafe { llvm::LLVMIsAConstantInt(v).map(|v| llvm::LLVMConstIntGetZExtValue(v)) }
    }

    fn const_to_opt_u128(&self, v: &'ll Value, sign_ext: bool) -> Option<u128> {
        unsafe {
            llvm::LLVMIsAConstantInt(v).and_then(|v| {
                let (mut lo, mut hi) = (0u64, 0u64);
                let success = llvm::LLVMRustConstInt128Get(v, sign_ext, &mut hi, &mut lo);
                success.then(|| hi_lo_to_u128(lo, hi))
            })
        }
    }

    fn scalar_to_backend(&self, cv: Scalar, layout: abi::Scalar, llty: &'ll Type) -> &'ll Value {
        trace!("Scalar to backend `{:?}`, `{:?}`, `{:?}`", cv, layout, llty);
        let bitsize = if layout.is_bool() {
            1
        } else {
            layout.value.size(self).bits()
        };
        let val = match cv {
            Scalar::Int(ScalarInt::ZST) => {
                assert_eq!(0, layout.value.size(self).bytes());
                self.const_undef(self.type_ix(0))
            }
            Scalar::Int(int) => {
                let data = int.assert_bits(layout.value.size(self));
                let llval = self.const_uint_big(self.type_ix(bitsize), data);
                if layout.value == Pointer {
                    unsafe { llvm::LLVMConstIntToPtr(llval, llty) }
                } else {
                    self.const_bitcast(llval, llty)
                }
            }
            Scalar::Ptr(ptr, _size) => {
                let (alloc_id, offset) = ptr.into_parts();
                let (base_addr, base_addr_space) = match self.tcx.global_alloc(alloc_id) {
                    GlobalAlloc::Memory(alloc) => {
                        let init = const_alloc_to_llvm(self, alloc);
                        let value = match alloc.mutability {
                            Mutability::Mut => self.static_addr_of_mut(init, alloc.align, None),
                            _ => self.static_addr_of(init, alloc.align, None),
                        };
                        if !self.sess().fewer_names() {
                            llvm::set_value_name(value, format!("{:?}", alloc_id).as_bytes());
                        }
                        (value, AddressSpace::DATA)
                    }
                    GlobalAlloc::Function(fn_instance) => (
                        self.get_fn_addr(fn_instance.polymorphize(self.tcx)),
                        self.data_layout().instruction_address_space,
                    ),
                    GlobalAlloc::Static(def_id) => {
                        assert!(self.tcx.is_static(def_id));
                        assert!(!self.tcx.is_thread_local_static(def_id));
                        (self.get_static(def_id), AddressSpace::DATA)
                    }
                };
                let llval = unsafe {
                    llvm::LLVMConstInBoundsGEP(
                        self.const_bitcast(base_addr, self.type_i8p_ext(base_addr_space)),
                        &self.const_usize(offset.bytes()),
                        1,
                    )
                };
                if layout.value != Pointer {
                    unsafe { llvm::LLVMConstPtrToInt(llval, llty) }
                } else {
                    self.const_bitcast(llval, llty)
                }
            }
        };

        trace!("...Scalar to backend: `{:?}`", val);

        val
    }

    fn from_const_alloc(
        &self,
        layout: TyAndLayout<'tcx>,
        alloc: &Allocation,
        offset: Size,
    ) -> PlaceRef<'tcx, &'ll Value> {
        assert_eq!(alloc.align, layout.align.abi);
        let llty = self.type_ptr_to(layout.llvm_type(self));
        let llval = if layout.size == Size::ZERO {
            let llval = self.const_usize(alloc.align.bytes());
            unsafe { llvm::LLVMConstIntToPtr(llval, llty) }
        } else {
            let init = const_alloc_to_llvm(self, alloc);
            let base_addr = self.static_addr_of(init, alloc.align, None);

            let llval = unsafe {
                llvm::LLVMConstInBoundsGEP(
                    self.const_bitcast(base_addr, self.type_i8p()),
                    &self.const_usize(offset.bytes()),
                    1,
                )
            };
            self.const_bitcast(llval, llty)
        };
        PlaceRef::new_sized(llval, layout)
    }

    fn const_ptrcast(&self, val: &'ll Value, ty: &'ll Type) -> &'ll Value {
        unsafe { llvm::LLVMConstPointerCast(val, ty) }
    }
}

#[inline]
fn hi_lo_to_u128(lo: u64, hi: u64) -> u128 {
    ((hi as u128) << 64) | (lo as u128)
}
