use std::borrow::Cow;

use libc::c_uint;
use rustc_abi::{FieldIdx, Size, TagEncoding, VariantIdx, Variants};
use rustc_codegen_ssa::debuginfo::{
    tag_base_type, type_names::compute_debuginfo_type_name, wants_c_like_enum_debuginfo,
};
use rustc_codegen_ssa::traits::{ConstCodegenMethods, MiscCodegenMethods};
use rustc_hir::def::CtorKind;
use rustc_index::IndexSlice;
use rustc_middle::bug;
use rustc_middle::mir::CoroutineLayout;
use rustc_middle::ty::layout::{LayoutOf, TyAndLayout};
use rustc_middle::ty::{self, AdtDef, CoroutineArgs, CoroutineArgsExt, Ty, VariantDef};
use rustc_span::Symbol;
use smallvec::smallvec;

use super::type_map::{DINodeCreationResult, UniqueTypeId};
use super::{SmallVec, size_and_align_of};
use crate::common::AsCCharPtr;
use crate::context::CodegenCx;
use crate::debug_info::file_metadata_from_def_id;
use crate::debug_info::metadata::type_map::{self, Stub, StubInfo};
use crate::debug_info::metadata::{
    NO_GENERICS, UNKNOWN_LINE_NUMBER, build_field_di_node, build_generic_type_param_di_nodes,
    file_metadata, type_di_node, unknown_file_metadata, visibility_di_flags,
};
use crate::debug_info::util::{DIB, create_DIArray, get_namespace_for_item};
use crate::llvm::debuginfo::{DIFile, DIFlags, DIType};
use crate::llvm::{self};

/// Build the debuginfo node for an enum type. The listing below shows how such a
/// type looks like at the LLVM IR/DWARF level. It is a `DW_TAG_structure_type`
/// with a single `DW_TAG_variant_part` that in turn contains a `DW_TAG_variant`
/// for each variant of the enum. The variant-part also contains a single member
/// describing the discriminant, and a nested struct type for each of the variants.
///
/// ```txt
///  ---> DW_TAG_structure_type              (top-level type for enum)
///         DW_TAG_variant_part              (variant part)
///           DW_AT_discr                    (reference to discriminant DW_TAG_member)
///           DW_TAG_member                  (discriminant member)
///           DW_TAG_variant                 (variant 1)
///           DW_TAG_variant                 (variant 2)
///           DW_TAG_variant                 (variant 3)
///         DW_TAG_structure_type            (type of variant 1)
///         DW_TAG_structure_type            (type of variant 2)
///         DW_TAG_structure_type            (type of variant 3)
/// ```
pub(super) fn build_enum_type_di_node<'ll, 'tcx>(
    cx: &CodegenCx<'ll, 'tcx>,
    unique_type_id: UniqueTypeId<'tcx>,
) -> DINodeCreationResult<'ll> {
    let enum_type = unique_type_id.expect_ty();
    let &ty::Adt(enum_adt_def, _) = enum_type.kind() else {
        bug!(
            "build_enum_type_di_node() called with non-enum type: `{:?}`",
            enum_type
        )
    };

    let enum_type_and_layout = cx.layout_of(enum_type);

    if wants_c_like_enum_debuginfo(cx.tcx, enum_type_and_layout) {
        return build_c_style_enum_di_node(cx, enum_adt_def, enum_type_and_layout);
    }

    let containing_scope = get_namespace_for_item(cx, enum_adt_def.did());
    let enum_type_name = compute_debuginfo_type_name(cx.tcx, enum_type, false);

    let visibility_flags = visibility_di_flags(cx, enum_adt_def.did(), enum_adt_def.did());

    let def_location = if cx.sess().opts.unstable_opts.debug_info_type_line_numbers {
        Some(file_metadata_from_def_id(cx, Some(enum_adt_def.did())))
    } else {
        None
    };

    type_map::build_type_with_children(
        cx,
        type_map::stub(
            cx,
            Stub::Struct,
            unique_type_id,
            &enum_type_name,
            def_location,
            size_and_align_of(enum_type_and_layout),
            Some(containing_scope),
            visibility_flags,
        ),
        |cx, enum_type_di_node| {
            // Build the struct type for each variant. These will be referenced by the
            // DW_TAG_variant DIEs inside of the DW_TAG_variant_part DIE.
            // We also called the names for the corresponding DW_TAG_variant DIEs here.
            let variant_member_infos: SmallVec<_> = enum_adt_def
                .variant_range()
                .map(|variant_index| VariantMemberInfo {
                    variant_index,
                    variant_name: Cow::from(enum_adt_def.variant(variant_index).name.as_str()),
                    variant_struct_type_di_node: build_enum_variant_struct_type_di_node(
                        cx,
                        enum_type_and_layout,
                        enum_type_di_node,
                        variant_index,
                        enum_adt_def.variant(variant_index),
                        enum_type_and_layout.for_variant(cx, variant_index),
                        visibility_flags,
                    ),
                    source_info: None,
                })
                .collect();

            let enum_adt_def_id = if cx.sess().opts.unstable_opts.debug_info_type_line_numbers {
                Some(enum_adt_def.did())
            } else {
                None
            };
            smallvec![build_enum_variant_part_di_node(
                cx,
                enum_type_and_layout,
                enum_type_di_node,
                enum_adt_def_id,
                &variant_member_infos[..],
            )]
        },
        // We don't seem to be emitting generic args on the enum type, it seems. Rather
        // they get attached to the struct type of each variant.
        NO_GENERICS,
    )
}

/// Build the debuginfo node for a coroutine environment. It looks the same as the debuginfo for
/// an enum. See [build_enum_type_di_node] for more information.
///
/// ```txt
///
///  ---> DW_TAG_structure_type              (top-level type for the coroutine)
///         DW_TAG_variant_part              (variant part)
///           DW_AT_discr                    (reference to discriminant DW_TAG_member)
///           DW_TAG_member                  (discriminant member)
///           DW_TAG_variant                 (variant 1)
///           DW_TAG_variant                 (variant 2)
///           DW_TAG_variant                 (variant 3)
///         DW_TAG_structure_type            (type of variant 1)
///         DW_TAG_structure_type            (type of variant 2)
///         DW_TAG_structure_type            (type of variant 3)
///
/// ```
pub(super) fn build_coroutine_di_node<'ll, 'tcx>(
    cx: &CodegenCx<'ll, 'tcx>,
    unique_type_id: UniqueTypeId<'tcx>,
) -> DINodeCreationResult<'ll> {
    let coroutine_type = unique_type_id.expect_ty();
    let &ty::Coroutine(coroutine_def_id, coroutine_args) = coroutine_type.kind() else {
        bug!(
            "build_coroutine_di_node() called with non-coroutine type: `{:?}`",
            coroutine_type
        )
    };

    let containing_scope = get_namespace_for_item(cx, coroutine_def_id);
    let coroutine_type_and_layout = cx.layout_of(coroutine_type);

    assert!(!wants_c_like_enum_debuginfo(
        cx.tcx,
        coroutine_type_and_layout
    ));

    let coroutine_type_name = compute_debuginfo_type_name(cx.tcx, coroutine_type, false);

    let def_location = if cx.sess().opts.unstable_opts.debug_info_type_line_numbers {
        Some(file_metadata_from_def_id(cx, Some(coroutine_def_id)))
    } else {
        None
    };

    type_map::build_type_with_children(
        cx,
        type_map::stub(
            cx,
            Stub::Struct,
            unique_type_id,
            &coroutine_type_name,
            def_location,
            size_and_align_of(coroutine_type_and_layout),
            Some(containing_scope),
            DIFlags::FlagZero,
        ),
        |cx, coroutine_type_di_node| {
            let coroutine_layout = cx
                .tcx
                .coroutine_layout(coroutine_def_id, coroutine_args.as_coroutine().kind_ty())
                .unwrap();

            let Variants::Multiple {
                tag_encoding: TagEncoding::Direct,
                ref variants,
                ..
            } = coroutine_type_and_layout.variants
            else {
                bug!(
                    "Encountered coroutine with non-direct-tag layout: {:?}",
                    coroutine_type_and_layout
                )
            };

            let common_upvar_names = cx
                .tcx
                .closure_saved_names_of_captured_variables(coroutine_def_id);

            // Build variant struct types
            let variant_struct_type_di_nodes: SmallVec<_> = variants
                .indices()
                .map(|variant_index| {
                    // FIXME: This is problematic because just a number is not a valid identifier.
                    //        CoroutineArgs::variant_name(variant_index), would be consistent
                    //        with enums?
                    let variant_name = format!("{}", variant_index.as_usize()).into();

                    let span = coroutine_layout.variant_source_info[variant_index].span;
                    let source_info = if !span.is_dummy() {
                        let loc = cx.lookup_debug_loc(span.lo());
                        Some((file_metadata(cx, &loc.file), loc.line))
                    } else {
                        None
                    };

                    VariantMemberInfo {
                        variant_index,
                        variant_name,
                        variant_struct_type_di_node: build_coroutine_variant_struct_type_di_node(
                            cx,
                            variant_index,
                            coroutine_type_and_layout,
                            coroutine_type_di_node,
                            coroutine_layout,
                            common_upvar_names,
                        ),
                        source_info,
                    }
                })
                .collect();

            let coroutine_def_id = if cx.sess().opts.unstable_opts.debug_info_type_line_numbers {
                Some(coroutine_def_id)
            } else {
                None
            };
            smallvec![build_enum_variant_part_di_node(
                cx,
                coroutine_type_and_layout,
                coroutine_type_di_node,
                coroutine_def_id,
                &variant_struct_type_di_nodes[..],
            )]
        },
        // We don't seem to be emitting generic args on the coroutine type, it seems. Rather
        // they get attached to the struct type of each variant.
        NO_GENERICS,
    )
}

/// Builds the DW_TAG_variant_part of an enum or coroutine debuginfo node:
///
/// ```txt
///       DW_TAG_structure_type              (top-level type for enum)
/// --->    DW_TAG_variant_part              (variant part)
///           DW_AT_discr                    (reference to discriminant DW_TAG_member)
///           DW_TAG_member                  (discriminant member)
///           DW_TAG_variant                 (variant 1)
///           DW_TAG_variant                 (variant 2)
///           DW_TAG_variant                 (variant 3)
///         DW_TAG_structure_type            (type of variant 1)
///         DW_TAG_structure_type            (type of variant 2)
///         DW_TAG_structure_type            (type of variant 3)
/// ```
fn build_enum_variant_part_di_node<'ll, 'tcx>(
    cx: &CodegenCx<'ll, 'tcx>,
    enum_type_and_layout: TyAndLayout<'tcx>,
    enum_type_di_node: &'ll DIType,
    enum_type_def_id: Option<rustc_span::def_id::DefId>,
    variant_member_infos: &[VariantMemberInfo<'_, 'll>],
) -> &'ll DIType {
    let tag_member_di_node =
        build_discr_member_di_node(cx, enum_type_and_layout, enum_type_di_node);

    let variant_part_unique_type_id =
        UniqueTypeId::for_enum_variant_part(cx.tcx, enum_type_and_layout.ty);

    let (file_metadata, line_number) = if cx.sess().opts.unstable_opts.debug_info_type_line_numbers
    {
        file_metadata_from_def_id(cx, enum_type_def_id)
    } else {
        (unknown_file_metadata(cx), UNKNOWN_LINE_NUMBER)
    };

    let stub = StubInfo::new(
        cx,
        variant_part_unique_type_id,
        |cx, variant_part_unique_type_id_str| unsafe {
            let variant_part_name = "";
            llvm::LLVMRustDIBuilderCreateVariantPart(
                DIB(cx),
                enum_type_di_node,
                variant_part_name.as_c_char_ptr(),
                variant_part_name.len(),
                file_metadata,
                line_number,
                enum_type_and_layout.size.bits(),
                enum_type_and_layout.align.abi.bits() as u32,
                DIFlags::FlagZero,
                tag_member_di_node,
                create_DIArray(DIB(cx), &[]),
                variant_part_unique_type_id_str.as_c_char_ptr(),
                variant_part_unique_type_id_str.len(),
            )
        },
    );

    type_map::build_type_with_children(
        cx,
        stub,
        |cx, variant_part_di_node| {
            variant_member_infos
                .iter()
                .map(|variant_member_info| {
                    build_enum_variant_member_di_node(
                        cx,
                        enum_type_and_layout,
                        variant_part_di_node,
                        variant_member_info,
                    )
                })
                .collect()
        },
        NO_GENERICS,
    )
    .di_node
}

/// Builds the DW_TAG_member describing where we can find the tag of an enum.
/// Returns `None` if the enum does not have a tag.
///
/// ```txt
///
///       DW_TAG_structure_type              (top-level type for enum)
///         DW_TAG_variant_part              (variant part)
///           DW_AT_discr                    (reference to discriminant DW_TAG_member)
/// --->      DW_TAG_member                  (discriminant member)
///           DW_TAG_variant                 (variant 1)
///           DW_TAG_variant                 (variant 2)
///           DW_TAG_variant                 (variant 3)
///         DW_TAG_structure_type            (type of variant 1)
///         DW_TAG_structure_type            (type of variant 2)
///         DW_TAG_structure_type            (type of variant 3)
///
/// ```
fn build_discr_member_di_node<'ll, 'tcx>(
    cx: &CodegenCx<'ll, 'tcx>,
    enum_or_coroutine_type_and_layout: TyAndLayout<'tcx>,
    enum_or_coroutine_type_di_node: &'ll DIType,
) -> Option<&'ll DIType> {
    let tag_name = match enum_or_coroutine_type_and_layout.ty.kind() {
        ty::Coroutine(..) => "__state",
        _ => "",
    };

    // NOTE: This is actually wrong. This will become a member of
    //       of the DW_TAG_variant_part. But, due to LLVM's API, that
    //       can only be constructed with this DW_TAG_member already in created.
    //       In LLVM IR the wrong scope will be listed but when DWARF is
    //       generated from it, the DW_TAG_member will be a child the
    //       DW_TAG_variant_part.
    let containing_scope = enum_or_coroutine_type_di_node;

    match enum_or_coroutine_type_and_layout.layout.variants() {
        // A single-variant or no-variant enum has no discriminant.
        &Variants::Single { .. } | &Variants::Empty => None,

        &Variants::Multiple { tag_field, .. } => {
            let tag_base_type = tag_base_type(cx.tcx, enum_or_coroutine_type_and_layout);
            let (size, align) = cx.size_and_align_of(tag_base_type);

            unsafe {
                Some(llvm::LLVMRustDIBuilderCreateMemberType(
                    DIB(cx),
                    containing_scope,
                    tag_name.as_c_char_ptr(),
                    tag_name.len(),
                    unknown_file_metadata(cx),
                    UNKNOWN_LINE_NUMBER,
                    size.bits(),
                    align.bits() as u32,
                    enum_or_coroutine_type_and_layout
                        .fields
                        .offset(tag_field)
                        .bits(),
                    DIFlags::FlagArtificial,
                    type_di_node(cx, tag_base_type),
                ))
            }
        }
    }
}

/// Build the debuginfo node for `DW_TAG_variant`:
///
/// ```txt
///       DW_TAG_structure_type              (top-level type for enum)
///         DW_TAG_variant_part              (variant part)
///           DW_AT_discr                    (reference to discriminant DW_TAG_member)
///           DW_TAG_member                  (discriminant member)
///  --->     DW_TAG_variant                 (variant 1)
///  --->     DW_TAG_variant                 (variant 2)
///  --->     DW_TAG_variant                 (variant 3)
///         DW_TAG_structure_type            (type of variant 1)
///         DW_TAG_structure_type            (type of variant 2)
///         DW_TAG_structure_type            (type of variant 3)
/// ```
///
/// This node looks like:
///
/// ```txt
/// DW_TAG_variant
///   DW_AT_discr_value           0
///   DW_TAG_member
///     DW_AT_name                  None
///     DW_AT_type                  <0x000002a1>
///     DW_AT_alignment             0x00000002
///     DW_AT_data_member_location  0
/// ```
///
/// The DW_AT_discr_value is optional, and is omitted if
///   - This is the only variant of a univariant enum (i.e. their is no discriminant)
///   - This is the "untagged" variant of a niche-layout enum
///     (where only the other variants are identified by a single value)
///
/// There is only ever a single member, the type of which is a struct that describes the
/// fields of the variant (excluding the discriminant). The name of the member is the name
/// of the variant as given in the source code. The DW_AT_data_member_location is always
/// zero.
///
/// Note that the LLVM DIBuilder API is a bit unintuitive here. The DW_TAG_variant subtree
/// (including the DW_TAG_member) is built by a single call to
/// `LLVMRustDIBuilderCreateVariantMemberType()`.
fn build_enum_variant_member_di_node<'ll, 'tcx>(
    cx: &CodegenCx<'ll, 'tcx>,
    enum_type_and_layout: TyAndLayout<'tcx>,
    variant_part_di_node: &'ll DIType,
    variant_member_info: &VariantMemberInfo<'_, 'll>,
) -> &'ll DIType {
    let variant_index = variant_member_info.variant_index;
    let discr_value = compute_discriminant_value(cx, enum_type_and_layout, variant_index);

    let (file_di_node, line_number) = variant_member_info
        .source_info
        .unwrap_or_else(|| (unknown_file_metadata(cx), UNKNOWN_LINE_NUMBER));

    let discr = discr_value.opt_single_val().map(|value| {
        let tag_base_type = tag_base_type(cx.tcx, enum_type_and_layout);
        let size = cx.size_of(tag_base_type);
        cx.const_uint_big(cx.type_ix(size.bits()), value)
    });

    unsafe {
        llvm::LLVMRustDIBuilderCreateVariantMemberType(
            DIB(cx),
            variant_part_di_node,
            variant_member_info.variant_name.as_c_char_ptr(),
            variant_member_info.variant_name.len(),
            file_di_node,
            line_number,
            enum_type_and_layout.size.bits(),
            enum_type_and_layout.align.abi.bits() as u32,
            Size::ZERO.bits(),
            discr,
            DIFlags::FlagZero,
            variant_member_info.variant_struct_type_di_node,
        )
    }
}

/// Information needed for building a `DW_TAG_variant`:
///
/// ```txt
///       DW_TAG_structure_type              (top-level type for enum)
///         DW_TAG_variant_part              (variant part)
///           DW_AT_discr                    (reference to discriminant DW_TAG_member)
///           DW_TAG_member                  (discriminant member)
///  --->     DW_TAG_variant                 (variant 1)
///  --->     DW_TAG_variant                 (variant 2)
///  --->     DW_TAG_variant                 (variant 3)
///         DW_TAG_structure_type            (type of variant 1)
///         DW_TAG_structure_type            (type of variant 2)
///         DW_TAG_structure_type            (type of variant 3)
/// ```
struct VariantMemberInfo<'a, 'll> {
    variant_index: VariantIdx,
    variant_name: Cow<'a, str>,
    variant_struct_type_di_node: &'ll DIType,
    source_info: Option<(&'ll DIFile, c_uint)>,
}

/// Build the debuginfo node for a C-style enum, i.e. an enum the variants of which have no fields.
///
/// The resulting debuginfo will be a DW_TAG_enumeration_type.
fn build_c_style_enum_di_node<'ll, 'tcx>(
    cx: &CodegenCx<'ll, 'tcx>,
    enum_adt_def: AdtDef<'tcx>,
    enum_type_and_layout: TyAndLayout<'tcx>,
) -> DINodeCreationResult<'ll> {
    let containing_scope = get_namespace_for_item(cx, enum_adt_def.did());
    let enum_adt_def_id = if cx.sess().opts.unstable_opts.debug_info_type_line_numbers {
        Some(enum_adt_def.did())
    } else {
        None
    };
    DINodeCreationResult {
        di_node: build_enumeration_type_di_node(
            cx,
            &compute_debuginfo_type_name(cx.tcx, enum_type_and_layout.ty, false),
            tag_base_type(cx.tcx, enum_type_and_layout),
            enum_adt_def
                .discriminants(cx.tcx)
                .map(|(variant_index, discr)| {
                    let name = Cow::from(enum_adt_def.variant(variant_index).name.as_str());
                    (name, discr.val)
                }),
            enum_adt_def_id,
            containing_scope,
        ),
        already_stored_in_typemap: false,
    }
}

/// Build a DW_TAG_enumeration_type debuginfo node, with the given base type and variants.
/// This is a helper function and does not register anything in the type map by itself.
///
/// `variants` is an iterator of (discr-value, variant-name).
///
/// NVVM: Discrimant values are mapped from u128 to i64 to conform with LLVM 7's API.
fn build_enumeration_type_di_node<'ll, 'tcx>(
    cx: &CodegenCx<'ll, 'tcx>,
    type_name: &str,
    base_type: Ty<'tcx>,
    enumerators: impl Iterator<Item = (Cow<'tcx, str>, u128)>,
    def_id: Option<rustc_span::def_id::DefId>,
    containing_scope: &'ll DIType,
) -> &'ll DIType {
    let is_unsigned = match base_type.kind() {
        ty::Int(_) => false,
        ty::Uint(_) => true,
        _ => bug!("build_enumeration_type_di_node() called with non-integer tag type."),
    };
    let (size, align) = cx.size_and_align_of(base_type);

    let enumerator_di_nodes: SmallVec<Option<&'ll DIType>> = enumerators
        .map(|(name, value)| unsafe {
            // FIXME: pass all 128 bits with newer LLVM API.
            Some(llvm::LLVMRustDIBuilderCreateEnumerator(
                DIB(cx),
                name.as_c_char_ptr(),
                name.len(),
                value as i64,
                is_unsigned
            ))
        })
        .collect();

    let (file_metadata, line_number) = if cx.sess().opts.unstable_opts.debug_info_type_line_numbers
    {
        file_metadata_from_def_id(cx, def_id)
    } else {
        (unknown_file_metadata(cx), UNKNOWN_LINE_NUMBER)
    };

    unsafe {
        llvm::LLVMRustDIBuilderCreateEnumerationType(
            DIB(cx),
            containing_scope,
            type_name.as_c_char_ptr(),
            type_name.len(),
            file_metadata,
            line_number,
            size.bits(),
            align.bits() as u32,
            create_DIArray(DIB(cx), &enumerator_di_nodes[..]),
            type_di_node(cx, base_type),
        )
    }
}

/// Build the debuginfo node for the struct type describing a single variant of an enum.
///
/// ```txt
///       DW_TAG_structure_type              (top-level type for enum)
///         DW_TAG_variant_part              (variant part)
///           DW_AT_discr                    (reference to discriminant DW_TAG_member)
///           DW_TAG_member                  (discriminant member)
///           DW_TAG_variant                 (variant 1)
///           DW_TAG_variant                 (variant 2)
///           DW_TAG_variant                 (variant 3)
///  --->   DW_TAG_structure_type            (type of variant 1)
///  --->   DW_TAG_structure_type            (type of variant 2)
///  --->   DW_TAG_structure_type            (type of variant 3)
/// ```
///
/// The node looks like:
///
/// ```txt
/// DW_TAG_structure_type
///   DW_AT_name                  <name-of-variant>
///   DW_AT_byte_size             0x00000010
///   DW_AT_alignment             0x00000008
///   DW_TAG_member
///     DW_AT_name                  <name-of-field-0>
///     DW_AT_type                  <0x0000018e>
///     DW_AT_alignment             0x00000004
///     DW_AT_data_member_location  4
///   DW_TAG_member
///     DW_AT_name                  <name-of-field-1>
///     DW_AT_type                  <0x00000195>
///     DW_AT_alignment             0x00000008
///     DW_AT_data_member_location  8
///   ...
/// ```
///
/// The type of a variant is always a struct type with the name of the variant
/// and a DW_TAG_member for each field (but not the discriminant).
fn build_enum_variant_struct_type_di_node<'ll, 'tcx>(
    cx: &CodegenCx<'ll, 'tcx>,
    enum_type_and_layout: TyAndLayout<'tcx>,
    enum_type_di_node: &'ll DIType,
    variant_index: VariantIdx,
    variant_def: &VariantDef,
    variant_layout: TyAndLayout<'tcx>,
    di_flags: DIFlags,
) -> &'ll DIType {
    assert_eq!(variant_layout.ty, enum_type_and_layout.ty);

    let def_location = if cx.sess().opts.unstable_opts.debug_info_type_line_numbers {
        Some(file_metadata_from_def_id(cx, Some(variant_def.def_id)))
    } else {
        None
    };

    type_map::build_type_with_children(
        cx,
        type_map::stub(
            cx,
            Stub::Struct,
            UniqueTypeId::for_enum_variant_struct_type(
                cx.tcx,
                enum_type_and_layout.ty,
                variant_index,
            ),
            variant_def.name.as_str(),
            def_location,
            // NOTE: We use size and align of enum_type, not from variant_layout:
            size_and_align_of(enum_type_and_layout),
            Some(enum_type_di_node),
            di_flags,
        ),
        |cx, struct_type_di_node| {
            (0..variant_layout.fields.count())
                .map(|field_index| {
                    let field_name = if variant_def.ctor_kind() != Some(CtorKind::Fn) {
                        // Fields have names
                        let field = &variant_def.fields[FieldIdx::from_usize(field_index)];
                        Cow::from(field.name.as_str())
                    } else {
                        // Tuple-like
                        super::tuple_field_name(field_index)
                    };

                    let field_layout = variant_layout.field(cx, field_index);

                    build_field_di_node(
                        cx,
                        struct_type_di_node,
                        &field_name,
                        (field_layout.size, field_layout.align.abi),
                        variant_layout.fields.offset(field_index),
                        di_flags,
                        type_di_node(cx, field_layout.ty),
                        None,
                    )
                })
                .collect::<SmallVec<_>>()
        },
        |cx| build_generic_type_param_di_nodes(cx, enum_type_and_layout.ty),
    )
    .di_node
}

/// Build the struct type for describing a single coroutine state.
/// See [build_coroutine_variant_struct_type_di_node].
///
/// ```txt
///
///       DW_TAG_structure_type              (top-level type for enum)
///         DW_TAG_variant_part              (variant part)
///           DW_AT_discr                    (reference to discriminant DW_TAG_member)
///           DW_TAG_member                  (discriminant member)
///           DW_TAG_variant                 (variant 1)
///           DW_TAG_variant                 (variant 2)
///           DW_TAG_variant                 (variant 3)
///  --->   DW_TAG_structure_type            (type of variant 1)
///  --->   DW_TAG_structure_type            (type of variant 2)
///  --->   DW_TAG_structure_type            (type of variant 3)
///
/// ```
fn build_coroutine_variant_struct_type_di_node<'ll, 'tcx>(
    cx: &CodegenCx<'ll, 'tcx>,
    variant_index: VariantIdx,
    coroutine_type_and_layout: TyAndLayout<'tcx>,
    coroutine_type_di_node: &'ll DIType,
    coroutine_layout: &CoroutineLayout<'tcx>,
    common_upvar_names: &IndexSlice<FieldIdx, Symbol>,
) -> &'ll DIType {
    let variant_name = CoroutineArgs::variant_name(variant_index);
    let unique_type_id = UniqueTypeId::for_enum_variant_struct_type(
        cx.tcx,
        coroutine_type_and_layout.ty,
        variant_index,
    );

    let variant_layout = coroutine_type_and_layout.for_variant(cx, variant_index);

    let coroutine_args = match coroutine_type_and_layout.ty.kind() {
        ty::Coroutine(_, args) => args.as_coroutine(),
        _ => unreachable!(),
    };

    type_map::build_type_with_children(
        cx,
        type_map::stub(
            cx,
            Stub::Struct,
            unique_type_id,
            &variant_name,
            None,
            size_and_align_of(coroutine_type_and_layout),
            Some(coroutine_type_di_node),
            DIFlags::FlagZero,
        ),
        |cx, variant_struct_type_di_node| {
            // Fields that just belong to this variant/state
            let state_specific_fields: SmallVec<_> = (0..variant_layout.fields.count())
                .map(|field_index| {
                    let coroutine_saved_local = coroutine_layout.variant_fields[variant_index]
                        [FieldIdx::from_usize(field_index)];
                    let field_name_maybe = coroutine_layout.field_names[coroutine_saved_local];
                    let field_name = field_name_maybe
                        .as_ref()
                        .map(|s| Cow::from(s.as_str()))
                        .unwrap_or_else(|| super::tuple_field_name(field_index));

                    let field_type = variant_layout.field(cx, field_index).ty;

                    build_field_di_node(
                        cx,
                        variant_struct_type_di_node,
                        &field_name,
                        cx.size_and_align_of(field_type),
                        variant_layout.fields.offset(field_index),
                        DIFlags::FlagZero,
                        type_di_node(cx, field_type),
                        None,
                    )
                })
                .collect();

            // Fields that are common to all states
            let common_fields: SmallVec<_> = coroutine_args
                .prefix_tys()
                .iter()
                .zip(common_upvar_names)
                .enumerate()
                .map(|(index, (upvar_ty, upvar_name))| {
                    build_field_di_node(
                        cx,
                        variant_struct_type_di_node,
                        upvar_name.as_str(),
                        cx.size_and_align_of(upvar_ty),
                        coroutine_type_and_layout.fields.offset(index),
                        DIFlags::FlagZero,
                        type_di_node(cx, upvar_ty),
                        None,
                    )
                })
                .collect();

            state_specific_fields
                .into_iter()
                .chain(common_fields)
                .collect()
        },
        |cx| build_generic_type_param_di_nodes(cx, coroutine_type_and_layout.ty),
    )
    .di_node
}

#[derive(Copy, Clone)]
enum DiscrResult {
    NoDiscriminant,
    Value(u128),
    #[allow(dead_code)]
    Range(u128, u128),
}

impl DiscrResult {
    fn opt_single_val(&self) -> Option<u128> {
        if let Self::Value(d) = *self {
            Some(d)
        } else {
            None
        }
    }
}

/// Returns the discriminant value corresponding to the variant index.
///
/// Will return `None` if there is less than two variants (because then the enum won't have)
/// a tag, and if this is the untagged variant of a niche-layout enum (because then there is no
/// single discriminant value).
fn compute_discriminant_value<'ll, 'tcx>(
    cx: &CodegenCx<'ll, 'tcx>,
    enum_type_and_layout: TyAndLayout<'tcx>,
    variant_index: VariantIdx,
) -> DiscrResult {
    match enum_type_and_layout.layout.variants() {
        &Variants::Single { .. } | &Variants::Empty => DiscrResult::NoDiscriminant,
        &Variants::Multiple {
            tag_encoding: TagEncoding::Direct,
            ..
        } => DiscrResult::Value(
            enum_type_and_layout
                .ty
                .discriminant_for_variant(cx.tcx, variant_index)
                .unwrap()
                .val,
        ),
        &Variants::Multiple {
            tag_encoding:
                TagEncoding::Niche {
                    ref niche_variants,
                    niche_start,
                    untagged_variant,
                },
            tag,
            ..
        } => {
            if variant_index == untagged_variant {
                let valid_range = enum_type_and_layout
                    .for_variant(cx, variant_index)
                    .largest_niche
                    .as_ref()
                    .unwrap()
                    .valid_range;

                let min = valid_range.start.min(valid_range.end);
                let min = tag.size(cx).truncate(min);

                let max = valid_range.start.max(valid_range.end);
                let max = tag.size(cx).truncate(max);

                DiscrResult::Range(min, max)
            } else {
                let value = (variant_index.as_u32() as u128)
                    .wrapping_sub(niche_variants.start().as_u32() as u128)
                    .wrapping_add(niche_start);
                let value = tag.size(cx).truncate(value);
                DiscrResult::Value(value)
            }
        }
    }
}
