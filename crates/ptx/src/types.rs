use smallvec::SmallVec;
use strum::EnumString;

#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub enum Directive {
    // module directives
    Version(VersionDirective),
    Target(TargetDirective),
    AddressSize(AddressSizeDirective),

    // function directives
    /// `.entry`
    Kernel(KernelDirective),
    Function(FunctionDirective),
    Alias(AliasDirective),

    // Control flow directives
    BranchTargets(BranchTargetsDirective),
    CallTargets(CallTargetsDirective),
    CallPrototype(CallPrototypeDirective),

    // performance tuning directives
    MaxNReg(MaxNRegDirective),
    MaxNTid(MaxNTidDirective),
    ReqNTid(ReqNTidDirective),
    MinNCtaPerSm(MinNCtaPerSmDirective),
    MaxNCtaPerSm(MaxNCtaPerSmDirective),
    Pragma(PragmaDirective),
    // debugging directives
    // not currently supported and unlikely to be in the future because its deprecated
    // and its syntax is not very well specified.
    // Dwarf(DwarfDirective)
    // Section(SectionDirective),
    // File(FileDirective),
    // Loc(LocDirective),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum DwarfLine {
    U8List(Vec<u8>),
    U16List(Vec<u16>),
    U32List(Vec<u32>),
    U64List(Vec<u64>),
    Label(String),
    U32LabelRef(String),
    U64LabelRef(String),
    U32LabelRefOffset(String, i32),
    U64LabelRefOffset(String, i64),
}

/// A directive used for declaring a section in the final cubin. Usually used
/// for DWARF debug info.
pub struct SectionDirective {
    pub name: String,
    pub lines: Vec<DwarfLine>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VersionDirective {
    pub major: u32,
    pub minor: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TargetDirective {
    pub specifiers: SmallVec<[TargetSpecifier; 5]>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, EnumString)]
#[strum(ascii_case_insensitive)]
pub enum TargetSpecifier {
    // target architectures
    Sm80,
    Sm86,

    Sm70,
    Sm72,
    Sm75,

    Sm60,
    Sm61,
    Sm62,

    Sm50,
    Sm52,
    Sm53,

    Sm30,
    Sm32,
    Sm35,
    Sm37,

    Sm20,

    Sm10,
    Sm11,
    Sm12,
    Sm13,

    // texturing modes
    #[strum(serialize = "texmode_unified")]
    TexmodeUnified,
    #[strum(serialize = "texmode_independent")]
    TexmodeIndependent,

    // platform options
    Debug,
    #[strum(serialize = "map_f64_to_f32")]
    MapF64ToF32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AddressSizeDirective {
    pub size: AddressSize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AddressSize {
    /// 32 bit
    Nvptx,
    /// 64 bit
    Nvptx64,
}

/// Declares a kernel function.
#[derive(Debug, Clone, PartialEq)]
pub struct KernelDirective {
    pub name: String,
    pub param_list: SmallVec<[(); 5]>,
    pub body: (),
}

/// Declares a general function.
#[derive(Debug, Clone, PartialEq)]
pub struct FunctionDirective {
    pub name: String,
    pub param_list: SmallVec<[(); 5]>,
    pub return_val: Option<()>,
    pub body: Option<()>,
}

/// Declares an alias of a function. Aliasee is the name of a function
/// prototype which matches the prototype of `alias`. Aliasee must not have weak linkage.
#[derive(Debug, Clone, PartialEq)]
pub struct AliasDirective {
    pub alias: String,
    pub aliasee: String,
}

/// Declares a list of potential branching targets for a `brx.idx` instructions, in the form
/// of a list of statement labels.
#[derive(Debug, Clone, PartialEq)]
pub struct BranchTargetsDirective {
    pub labels: Vec<String>,
}

/// Declares a list of potential call targets for an indirect call instruction. And associates
/// the list with the label at the start of the line.
#[derive(Debug, Clone, PartialEq)]
pub struct CallTargetsDirective {
    pub functions: Vec<String>,
}

/// Defines a prototype without a specific function name, and associates that prototype with a label.
/// The prototype can then be used in indirect calls where there is incomplete knowledge of the possible
/// call targets.
#[derive(Debug, Clone, PartialEq)]
pub struct CallPrototypeDirective {
    pub inputs: Vec<()>,
    pub return_params: Vec<()>,
}

/// Declares the max number of registers that can be allocated per thread.
#[derive(Debug, Clone, PartialEq)]
pub struct MaxNRegDirective {
    pub max: u32,
}

/// Declares the max number of threads allowed in a thread block for a kernel.
#[derive(Debug, Clone, PartialEq)]
pub struct MaxNTidDirective {
    pub max_x: u32,
    pub max_y: Option<u32>,
    pub max_z: Option<u32>,
}

/// Declares that a kernel with this directive must only be launched with the thread block dimensions
/// specified in this directive.
#[derive(Debug, Clone, PartialEq)]
pub struct ReqNTidDirective {
    pub x: u32,
    pub y: Option<u32>,
    pub z: Option<u32>,
}

/// Declares the minimum number of thread blocks from a kernel's grid to be mapped to a single
/// multiprocessor.
///
/// This directive generally requires `.maxntid` or `.reqntid` to be specified as well.
#[derive(Debug, Clone, PartialEq)]
pub struct MinNCtaPerSmDirective {
    pub min: u32,
}

/// (deprecated) Declares the maximum number of thread blocks from a kernel's grid to be mapped to a single
/// multiprocessor.
///
/// This directive generally requires `.maxntid` or `.reqntid` to be specified as well.
#[derive(Debug, Clone, PartialEq)]
pub struct MaxNCtaPerSmDirective {
    pub max: u32,
}

/// A directive to pass directives to the PTX compiler. This can either be module-scoped, entry-scoped, or
/// statement-scoped.
///
/// Currently the only recognized pragma string is `unroll`.
#[derive(Debug, Clone, PartialEq)]
pub struct PragmaDirective {
    pub pragma: String,
}

// ------------------------------------------------------
// Token types
// ------------------------------------------------------

#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TokenKind {
    Directive,
    Instruction,
    Option,

    Ident,

    ParenOpen,
    ParenClose,
    SquareBracketOpen,
    SquareBracketClose,
    CurlyBracketOpen,
    CurlyBracketClose,

    Semicolon,
    Colon,
    Dot,
    Plus,
    Minus,
    Bang,
    Tilde,
    Times,
    Divide,
    Modulo,
    LeftShift,
    RightShift,
    LessThan,
    LessThanOrEqualTo,
    GreaterThan,
    GreaterThanOrEqualTo,
    And,
    Xor,
    Or,
    LogicalAnd,
    LogicalOr,
    QuestionMark,
    Equals,
    NotEquals,
    At,
    Comma,
    Assign,

    String,

    UnsignedInt,
    SignedInt,
    Float,
    Double,

    Type,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TokenValue {
    Directive(DirectiveKind),
    Instruction(InstructionKind),
    Option(InstructionOption),
    UnsignedInt(u64),
    SignedInt(i64),
    Float(f32),
    Double(f64),
    Type(ReservedType),
    Ident(String),
}

/// A type of directive, see [nvidia docs](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#directive-statements).
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, EnumString)]
#[strum(ascii_case_insensitive)]
pub enum DirectiveKind {
    // this one is different for... reasons...
    #[strum(serialize = "address_size")]
    AddressSize,
    Align,
    BranchTargets,
    CallPrototype,
    CallTargets,
    Const,
    Entry,
    Extern,
    File,
    Func,
    Global,
    Loc,
    Local,
    MaxNCtaPerSm,
    MaxNReg,
    MaxNTid,
    MinNCtaPerSm,
    Param,
    Pragma,
    Reg,
    ReqNTid,
    Section,
    Shared,
    Sreg,
    Target,
    Tex,
    Version,
    Visible,
    Weak,
    // this one is not mentioned in the directive table but its
    // used in function definitions so include it as one.
    NoReturn,
}

/// A type of instruction, see [nvidia docs](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#instruction-statements__reserved-instruction-keywords)
#[repr(u8)]
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, EnumString)]
#[strum(ascii_case_insensitive)]
pub enum InstructionKind {
    Abs,
    Activemask,
    Add,
    Addc,
    Alloca,
    And,
    Applypriority,
    Atom,
    Bar,
    Barrier,
    Bfe,
    Bfi,
    Bfind,
    Bra,
    Brev,
    Brkpt,
    Brx,
    Call,
    Clz,
    Cnot,
    Copysign,
    Cos,
    Cp,
    Createpolicy,
    Cvta,
    Discard,
    Div,
    Dp2a,
    Dp4a,
    Ex2,
    Exit,
    Fence,
    Fma,
    Fns,
    Isspacep,
    Istypep,
    Ld,
    Ldmatrix,
    Ldu,
    Lg2,
    Lop3,
    Mad,
    Mad24,
    Madc,
    Match,
    Max,
    Mbarrier,
    Membar,
    Min,
    Mma,
    Mov,
    Mul,
    Mul24,
    Nanosleep,
    Neg,
    Not,
    Or,
    Pmevent,
    Popc,
    Prefetch,
    Prefetchu,
    Prmt,
    Rcp,
    Red,
    Redux,
    Rem,
    Ret,
    Rsqrt,
    Sad, // :(
    Selp,
    Set,
    Setp,
    Shf,
    Shfl,
    Shl,
    Shr,
    Sin,
    Slct,
    Sqrt,
    St,
    Stackrestore,
    Stacksave,
    Sub,
    Subc,
    Suld,
    Suq,
    Sured,
    Sust,
    Tanh,
    Testp,
    Tex,
    Tld4,
    Trap,
    Txq,
    Vabsdiff,
    Vabsdiff2,
    Vabsdiff4,
    Vadd,
    Vadd2,
    Vadd4,
    Vavrg2,
    Vavrg4,
    Vmad,
    Vmax,
    Vmax2,
    Vmax4,
    Vmin,
    Vmin2,
    Vmin4,
    Vote,
    Vset,
    Vset2,
    Vset4,
    Vshl,
    Vshr,
    Vsub,
    Vsub2,
    Vsub4,
    Wmma,
    Xor,
}

/// An option for an instruction.
///
/// Taken from [gpgpu-sim_distribution](https://github.com/gpgpu-sim/gpgpu-sim_distribution/blob/master/cuobjdump_to_ptxplus/ptx.l) because
/// it seems nvidia just forgot to specify what these could be.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, EnumString)]
#[non_exhaustive]
#[strum(ascii_case_insensitive)]
pub enum InstructionOption {
    Row,
    Col,
    M16N16K16,
    F4e,
    B4e,
    Rc8,
    Ecl,
    Ecr,
    Rc16,

    Equ,
    Neu,
    Ltu,
    Leu,
    Gtu,
    Geu,
    Num,
    Nan,

    Sat,

    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    Cf,
    Sf,
    Nsf,

    Lo,
    Ls,
    Hi,
    Hs,

    Rni,
    Rzi,
    Rmi,
    Rpi,

    Rn,
    Rz,
    Rm,
    Rp,

    Ftz,

    Neg,

    Wide,
    Uni,

    Sync,
    Arrive,
    Red,

    Approx,
    Full,

    Any,
    All,
    Ballot,
    Gl,
    Cta,
    Sys,

    Exit,

    Abs,

    To,

    Ca,
    Cg,
    Cs,
    Lu,
    Cv,

    Wb,
    Wt,

    Nc,

    Up,
    Down,
    Bfly,
    Idx,

    Popc,
    And,
    Or,
    Xor,
    Cas,
    Exch,
    Add,
    Inc,
    Dec,
    Min,
    Max,

    #[strum(serialize = "1d")]
    _1d,
    #[strum(serialize = "2d")]
    _2d,
    #[strum(serialize = "3d")]
    _3d,
    #[strum(serialize = "0")]
    Dim0,
    #[strum(serialize = "1")]
    Dim1,
    #[strum(serialize = "2")]
    Dim2,
    #[strum(serialize = "x")]
    DimX,
    #[strum(serialize = "y")]
    DimY,
    #[strum(serialize = "z")]
    DimZ,

    Warp,

    Load,
    Store,

    Volatile,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, EnumString)]
#[strum(ascii_case_insensitive)]
pub enum ReservedType {
    S8,
    S16,
    S32,
    S64,
    U8,
    U16,
    U32,
    U64,
    F16,
    F16x2,
    F32,
    F64,
    FF64,
    B8,
    B16,
    B32,
    B64,
    BB64,
    BB128,
    Pred,

    TexRef,
    SamplerRef,
    SurfRef,

    V2,
    V3,
    V4,
}
