use crate::rt::sys::cudaError_t;
use core::result::Result;

#[repr(i32)]
#[non_exhaustive]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum CudaError {
    InvalidValue = 1,
    MemoryAllocation = 2,
    InitializationError = 3,
    CudartUnloading = 4,
    ProfilerDisabled = 5,
    ProfilerNotInitialized = 6,
    ProfilerAlreadyStarted = 7,
    ProfilerAlreadyStopped = 8,
    InvalidConfiguration = 9,
    InvalidPitchValue = 12,
    InvalidSymbol = 13,
    InvalidHostPointer = 16,
    InvalidDevicePointer = 17,
    InvalidTexture = 18,
    InvalidTextureBinding = 19,
    InvalidChannelDescriptor = 20,
    InvalidMemcpyDirection = 21,
    AddressOfConstant = 22,
    TextureFetchFailed = 23,
    TextureNotBound = 24,
    SynchronizationError = 25,
    InvalidFilterSetting = 26,
    InvalidNormSetting = 27,
    MixedDeviceExecution = 28,
    NotYetImplemented = 31,
    MemoryValueTooLarge = 32,
    StubLibrary = 34,
    InsufficientDriver = 35,
    CallRequiresNewerDriver = 36,
    InvalidSurface = 37,
    DuplicateVariableName = 43,
    DuplicateTextureName = 44,
    DuplicateSurfaceName = 45,
    DevicesUnavailable = 46,
    IncompatibleDriverContext = 49,
    MissingConfiguration = 52,
    PriorLaunchFailure = 53,
    LaunchMaxDepthExceeded = 65,
    LaunchFileScopedTex = 66,
    LaunchFileScopedSurf = 67,
    SyncDepthExceeded = 68,
    LaunchPendingCountExceeded = 69,
    InvalidDeviceFunction = 98,
    NoDevice = 100,
    InvalidDevice = 101,
    DeviceNotLicensed = 102,
    SoftwareValidityNotEstablished = 103,
    StartupFailure = 127,
    InvalidKernelImage = 200,
    DeviceUninitialized = 201,
    MapBufferObjectFailed = 205,
    UnmapBufferObjectFailed = 206,
    ArrayIsMapped = 207,
    AlreadyMapped = 208,
    NoKernelImageForDevice = 209,
    AlreadyAcquired = 210,
    NotMapped = 211,
    NotMappedAsArray = 212,
    NotMappedAsPointer = 213,
    ECCUncorrectable = 214,
    UnsupportedLimit = 215,
    DeviceAlreadyInUse = 216,
    PeerAccessUnsupported = 217,
    InvalidPtx = 218,
    InvalidGraphicsContext = 219,
    NvlinkUncorrectable = 220,
    JitCompilerNotFound = 221,
    UnsupportedPtxVersion = 222,
    JitCompilationDisabled = 223,
    UnsupportedExecAffinity = 224,
    InvalidSource = 300,
    FileNotFound = 301,
    SharedObjectSymbolNotFound = 302,
    SharedObjectInitFailed = 303,
    OperatingSystem = 304,
    InvalidResourceHandle = 400,
    IllegalState = 401,
    SymbolNotFound = 500,
    NotReady = 600,
    IllegalAddress = 700,
    LaunchOutOfResources = 701,
    LaunchTimeout = 702,
    LaunchIncompatibleTexturing = 703,
    PeerAccessAlreadyEnabled = 704,
    PeerAccessNotEnabled = 705,
    SetOnActiveProcess = 708,
    ContextIsDestroyed = 709,
    Assert = 710,
    TooManyPeers = 711,
    HostMemoryAlreadyRegistered = 712,
    HostMemoryNotRegistered = 713,
    HardwareStackError = 714,
    IllegalInstruction = 715,
    MisalignedAddress = 716,
    InvalidAddressSpace = 717,
    InvalidPc = 718,
    LaunchFailure = 719,
    CooperativeLaunchTooLarge = 720,
    NotPermitted = 800,
    NotSupported = 801,
    SystemNotReady = 802,
    SystemDriverMismatch = 803,
    CompatNotSupportedOnDevice = 804,
    MpsConnectionFailed = 805,
    MpsRpcFailure = 806,
    MpsServerNotReady = 807,
    MpsMaxClientsReached = 808,
    MpsMaxConnectionsReached = 809,
    StreamCaptureUnsupported = 900,
    StreamCaptureInvalidated = 901,
    StreamCaptureMerge = 902,
    StreamCaptureUnmatched = 903,
    StreamCaptureUnjoined = 904,
    StreamCaptureIsolation = 905,
    StreamCaptureImplicit = 906,
    CapturedEvent = 907,
    StreamCaptureWrongThread = 908,
    Timeout = 909,
    GraphExecUpdateFailure = 910,
    ExternalDevice = 911,
    Unknown = 999,
    ApiFailureBase = 10000,
}

/// Result type for most CUDA functions.
pub type CudaResult<T> = Result<T, CudaError>;

pub(crate) trait ToResult {
    fn to_result(self) -> CudaResult<()>;
}
impl ToResult for cudaError_t {
    fn to_result(self) -> CudaResult<()> {
        match self {
            Self::cudaSuccess => Ok(()),
            // SAFETY: cudaError_t and CudaError use the same integer reprs.
            _ => unsafe { Err(core::mem::transmute::<_, CudaError>(self)) },
        }
    }
}