use crate::{error::OptixResult, optix_call, sys};

#[derive(Clone)]
#[repr(transparent)]
pub struct Module {
    pub(crate) raw: sys::OptixModule,
}

#[repr(u32)]
#[derive(Debug, Hash, PartialEq, Copy, Clone)]
pub enum CompileOptimizationLevel {
    Default = sys::OptixCompileOptimizationLevel::OPTIX_COMPILE_OPTIMIZATION_DEFAULT,
    Level0 = sys::OptixCompileOptimizationLevel::OPTIX_COMPILE_OPTIMIZATION_LEVEL_0,
    Level1 = sys::OptixCompileOptimizationLevel::OPTIX_COMPILE_OPTIMIZATION_LEVEL_1,
    Level2 = sys::OptixCompileOptimizationLevel::OPTIX_COMPILE_OPTIMIZATION_LEVEL_2,
    Level3 = sys::OptixCompileOptimizationLevel::OPTIX_COMPILE_OPTIMIZATION_LEVEL_3,
}

#[repr(u32)]
#[derive(Debug, Hash, PartialEq, Copy, Clone)]
pub enum CompileDebugLevel {
    None = sys::OptixCompileDebugLevel::OPTIX_COMPILE_DEBUG_LEVEL_NONE,
    LineInfo = sys::OptixCompileDebugLevel::OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO,
    Full = sys::OptixCompileDebugLevel::OPTIX_COMPILE_DEBUG_LEVEL_FULL,
}
