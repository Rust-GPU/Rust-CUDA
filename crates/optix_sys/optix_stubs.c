/*
 * Copyright (c) 2021 NVIDIA Corporation.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/// @file
/// @author NVIDIA Corporation
/// @brief  OptiX public API header

#include "optix_function_table.h"

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN 1
#endif
#include <windows.h>
// The cfgmgr32 header is necessary for interrogating driver information in the registry.
// For convenience the library is also linked in automatically using the #pragma command.
#include <cfgmgr32.h>
#pragma comment(lib, "Cfgmgr32.lib")
#include <string.h>
#else
#include <dlfcn.h>
#endif

#ifdef __cplusplus
extern "C"
{
#endif

// The function table needs to be defined in exactly one translation unit. This can be
// achieved by including optix_function_table_definition.h in that translation unit.
// extern OptixFunctionTable g_optixFunctionTable;
#include "optix_function_table_definition.h"

#ifdef _WIN32
  static void *optixLoadWindowsDllFromName(const char *optixDllName)
  {
    void *handle = NULL;

    // Get the size of the path first, then allocate
    unsigned int size = GetSystemDirectoryA(NULL, 0);
    if (size == 0)
    {
      // Couldn't get the system path size, so bail
      return NULL;
    }
    size_t pathSize = size + 1 + strlen(optixDllName);
    char *systemPath = (char *)malloc(pathSize);
    if (systemPath == NULL)
      return NULL;
    if (GetSystemDirectoryA(systemPath, size) != size - 1)
    {
      // Something went wrong
      free(systemPath);
      return NULL;
    }
    strcat(systemPath, "\\");
    strcat(systemPath, optixDllName);
    handle = LoadLibraryA(systemPath);
    free(systemPath);
    if (handle)
      return handle;

    // If we didn't find it, go looking in the register store.  Since nvoptix.dll doesn't
    // have its own registry entry, we are going to look for the opengl driver which lives
    // next to nvoptix.dll.  0 (null) will be returned if any errors occured.

    static const char *deviceInstanceIdentifiersGUID = "{4d36e968-e325-11ce-bfc1-08002be10318}";
    const ULONG flags = CM_GETIDLIST_FILTER_CLASS | CM_GETIDLIST_FILTER_PRESENT;
    ULONG deviceListSize = 0;
    if (CM_Get_Device_ID_List_SizeA(&deviceListSize, deviceInstanceIdentifiersGUID, flags) != CR_SUCCESS)
    {
      return NULL;
    }
    char *deviceNames = (char *)malloc(deviceListSize);
    if (deviceNames == NULL)
      return NULL;
    if (CM_Get_Device_ID_ListA(deviceInstanceIdentifiersGUID, deviceNames, deviceListSize, flags))
    {
      free(deviceNames);
      return NULL;
    }
    DEVINST devID = 0;
    char *dllPath = NULL;

    // Continue to the next device if errors are encountered.
    for (char *deviceName = deviceNames; *deviceName; deviceName += strlen(deviceName) + 1)
    {
      if (CM_Locate_DevNodeA(&devID, deviceName, CM_LOCATE_DEVNODE_NORMAL) != CR_SUCCESS)
      {
        continue;
      }
      HKEY regKey = 0;
      if (CM_Open_DevNode_Key(devID, KEY_QUERY_VALUE, 0, RegDisposition_OpenExisting, &regKey, CM_REGISTRY_SOFTWARE) != CR_SUCCESS)
      {
        continue;
      }
      const char *valueName = "OpenGLDriverName";
      DWORD valueSize = 0;
      LSTATUS ret = RegQueryValueExA(regKey, valueName, NULL, NULL, NULL, &valueSize);
      if (ret != ERROR_SUCCESS)
      {
        RegCloseKey(regKey);
        continue;
      }
      char *regValue = (char *)malloc(valueSize);
      if (regValue == NULL)
      {
        RegCloseKey(regKey);
        continue;
      }
      ret = RegQueryValueExA(regKey, valueName, NULL, NULL, (LPBYTE)regValue, &valueSize);
      if (ret != ERROR_SUCCESS)
      {
        free(regValue);
        RegCloseKey(regKey);
        continue;
      }
      // Strip the opengl driver dll name from the string then create a new string with
      // the path and the nvoptix.dll name
      for (int i = (int)valueSize - 1; i >= 0 && regValue[i] != '\\'; --i)
        regValue[i] = '\0';
      size_t newPathSize = strlen(regValue) + strlen(optixDllName) + 1;
      dllPath = (char *)malloc(newPathSize);
      if (dllPath == NULL)
      {
        free(regValue);
        RegCloseKey(regKey);
        continue;
      }
      strcpy(dllPath, regValue);
      strcat(dllPath, optixDllName);
      free(regValue);
      RegCloseKey(regKey);
      handle = LoadLibraryA((LPCSTR)dllPath);
      free(dllPath);
      if (handle)
        break;
    }
    free(deviceNames);
    return handle;
  }

  static void *optixLoadWindowsDll()
  {
    return optixLoadWindowsDllFromName("nvoptix.dll");
  }
#endif

  /// \defgroup optix_utilities Utilities
  /// \brief OptiX Utilities

  /** \addtogroup optix_utilities
@{
*/

  /// Loads the OptiX library and initializes the function table used by the stubs below.
  ///
  /// If handlePtr is not nullptr, an OS-specific handle to the library will be returned in *handlePtr.
  ///
  /// \see #optixUninitWithHandle
  OptixResult optixInitWithHandle(void **handlePtr)
  {
    // Make sure these functions get initialized to zero in case the DLL and function
    // table can't be loaded
    g_optixFunctionTable.optixGetErrorName = 0;
    g_optixFunctionTable.optixGetErrorString = 0;

    if (!handlePtr)
      return OPTIX_ERROR_INVALID_VALUE;

#ifdef _WIN32
    *handlePtr = optixLoadWindowsDll();
    if (!*handlePtr)
      return OPTIX_ERROR_LIBRARY_NOT_FOUND;

    void *symbol = GetProcAddress((HMODULE)*handlePtr, "optixQueryFunctionTable");
    if (!symbol)
      return OPTIX_ERROR_ENTRY_SYMBOL_NOT_FOUND;
#else
  *handlePtr = dlopen("libnvoptix.so.1", RTLD_NOW);
  if (!*handlePtr)
    return OPTIX_ERROR_LIBRARY_NOT_FOUND;

  void *symbol = dlsym(*handlePtr, "optixQueryFunctionTable");
  if (!symbol)
    return OPTIX_ERROR_ENTRY_SYMBOL_NOT_FOUND;
#endif

    OptixQueryFunctionTable_t *optixQueryFunctionTable = (OptixQueryFunctionTable_t *)symbol;

    return optixQueryFunctionTable(OPTIX_ABI_VERSION, 0, 0, 0, &g_optixFunctionTable, sizeof(g_optixFunctionTable));
  }

  /// Loads the OptiX library and initializes the function table used by the stubs below.
  ///
  /// A variant of #optixInitWithHandle() that does not make the handle to the loaded library available.
  OptixResult optixInit(void)
  {
    void *handle;
    return optixInitWithHandle(&handle);
  }

  /// Unloads the OptiX library and zeros the function table used by the stubs below.  Takes the
  /// handle returned by optixInitWithHandle.  All OptixDeviceContext objects must be destroyed
  /// before calling this function, or the behavior is undefined.
  ///
  /// \see #optixInitWithHandle
  OptixResult optixUninitWithHandle(void *handle)
  {
    if (!handle)
      return OPTIX_ERROR_INVALID_VALUE;
#ifdef _WIN32
    if (!FreeLibrary((HMODULE)handle))
      return OPTIX_ERROR_LIBRARY_UNLOAD_FAILURE;
#else
  if (dlclose(handle))
    return OPTIX_ERROR_LIBRARY_UNLOAD_FAILURE;
#endif
    OptixFunctionTable empty = {0};
    g_optixFunctionTable = empty;
    return OPTIX_SUCCESS;
  }

  /*@}*/ // end group optix_utilities

  // Stub functions that forward calls to the corresponding function pointer in the function table.

  const char *optixGetErrorName(OptixResult result)
  {
    if (g_optixFunctionTable.optixGetErrorName)
      return g_optixFunctionTable.optixGetErrorName(result);

    // If the DLL and symbol table couldn't be loaded, provide a set of error strings
    // suitable for processing errors related to the DLL loading.
    switch (result)
    {
    case OPTIX_SUCCESS:
      return "OPTIX_SUCCESS";
    case OPTIX_ERROR_INVALID_VALUE:
      return "OPTIX_ERROR_INVALID_VALUE";
    case OPTIX_ERROR_UNSUPPORTED_ABI_VERSION:
      return "OPTIX_ERROR_UNSUPPORTED_ABI_VERSION";
    case OPTIX_ERROR_FUNCTION_TABLE_SIZE_MISMATCH:
      return "OPTIX_ERROR_FUNCTION_TABLE_SIZE_MISMATCH";
    case OPTIX_ERROR_INVALID_ENTRY_FUNCTION_OPTIONS:
      return "OPTIX_ERROR_INVALID_ENTRY_FUNCTION_OPTIONS";
    case OPTIX_ERROR_LIBRARY_NOT_FOUND:
      return "OPTIX_ERROR_LIBRARY_NOT_FOUND";
    case OPTIX_ERROR_ENTRY_SYMBOL_NOT_FOUND:
      return "OPTIX_ERROR_ENTRY_SYMBOL_NOT_FOUND";
    case OPTIX_ERROR_LIBRARY_UNLOAD_FAILURE:
      return "OPTIX_ERROR_LIBRARY_UNLOAD_FAILURE";
    default:
      return "Unknown OptixResult code";
    }
  }

  const char *optixGetErrorString(OptixResult result)
  {
    if (g_optixFunctionTable.optixGetErrorString)
      return g_optixFunctionTable.optixGetErrorString(result);

    // If the DLL and symbol table couldn't be loaded, provide a set of error strings
    // suitable for processing errors related to the DLL loading.
    switch (result)
    {
    case OPTIX_SUCCESS:
      return "Success";
    case OPTIX_ERROR_INVALID_VALUE:
      return "Invalid value";
    case OPTIX_ERROR_UNSUPPORTED_ABI_VERSION:
      return "Unsupported ABI version";
    case OPTIX_ERROR_FUNCTION_TABLE_SIZE_MISMATCH:
      return "Function table size mismatch";
    case OPTIX_ERROR_INVALID_ENTRY_FUNCTION_OPTIONS:
      return "Invalid options to entry function";
    case OPTIX_ERROR_LIBRARY_NOT_FOUND:
      return "Library not found";
    case OPTIX_ERROR_ENTRY_SYMBOL_NOT_FOUND:
      return "Entry symbol not found";
    case OPTIX_ERROR_LIBRARY_UNLOAD_FAILURE:
      return "Library could not be unloaded";
    default:
      return "Unknown OptixResult code";
    }
  }

  OptixResult optixDeviceContextCreate(CUcontext fromContext, const OptixDeviceContextOptions *options, OptixDeviceContext *context)
  {
    return g_optixFunctionTable.optixDeviceContextCreate(fromContext, options, context);
  }

  OptixResult optixDeviceContextDestroy(OptixDeviceContext context)
  {
    return g_optixFunctionTable.optixDeviceContextDestroy(context);
  }

  OptixResult optixDeviceContextGetProperty(OptixDeviceContext context, OptixDeviceProperty property, void *value, size_t sizeInBytes)
  {
    return g_optixFunctionTable.optixDeviceContextGetProperty(context, property, value, sizeInBytes);
  }

  OptixResult optixDeviceContextSetLogCallback(OptixDeviceContext context,
                                               OptixLogCallback callbackFunction,
                                               void *callbackData,
                                               unsigned int callbackLevel)
  {
    return g_optixFunctionTable.optixDeviceContextSetLogCallback(context, callbackFunction, callbackData, callbackLevel);
  }

  OptixResult optixDeviceContextSetCacheEnabled(OptixDeviceContext context, int enabled)
  {
    return g_optixFunctionTable.optixDeviceContextSetCacheEnabled(context, enabled);
  }

  OptixResult optixDeviceContextSetCacheLocation(OptixDeviceContext context, const char *location)
  {
    return g_optixFunctionTable.optixDeviceContextSetCacheLocation(context, location);
  }

  OptixResult optixDeviceContextSetCacheDatabaseSizes(OptixDeviceContext context, size_t lowWaterMark, size_t highWaterMark)
  {
    return g_optixFunctionTable.optixDeviceContextSetCacheDatabaseSizes(context, lowWaterMark, highWaterMark);
  }

  OptixResult optixDeviceContextGetCacheEnabled(OptixDeviceContext context, int *enabled)
  {
    return g_optixFunctionTable.optixDeviceContextGetCacheEnabled(context, enabled);
  }

  OptixResult optixDeviceContextGetCacheLocation(OptixDeviceContext context, char *location, size_t locationSize)
  {
    return g_optixFunctionTable.optixDeviceContextGetCacheLocation(context, location, locationSize);
  }

  OptixResult optixDeviceContextGetCacheDatabaseSizes(OptixDeviceContext context, size_t *lowWaterMark, size_t *highWaterMark)
  {
    return g_optixFunctionTable.optixDeviceContextGetCacheDatabaseSizes(context, lowWaterMark, highWaterMark);
  }

  OptixResult optixModuleCreateFromPTX(OptixDeviceContext context,
                                       const OptixModuleCompileOptions *moduleCompileOptions,
                                       const OptixPipelineCompileOptions *pipelineCompileOptions,
                                       const char *PTX,
                                       size_t PTXsize,
                                       char *logString,
                                       size_t *logStringSize,
                                       OptixModule *module)
  {
    return g_optixFunctionTable.optixModuleCreateFromPTX(context, moduleCompileOptions, pipelineCompileOptions, PTX,
                                                         PTXsize, logString, logStringSize, module);
  }

  OptixResult optixModuleDestroy(OptixModule module)
  {
    return g_optixFunctionTable.optixModuleDestroy(module);
  }

  OptixResult optixBuiltinISModuleGet(OptixDeviceContext context,
                                      const OptixModuleCompileOptions *moduleCompileOptions,
                                      const OptixPipelineCompileOptions *pipelineCompileOptions,
                                      const OptixBuiltinISOptions *builtinISOptions,
                                      OptixModule *builtinModule)
  {
    return g_optixFunctionTable.optixBuiltinISModuleGet(context, moduleCompileOptions, pipelineCompileOptions,
                                                        builtinISOptions, builtinModule);
  }

  OptixResult optixProgramGroupCreate(OptixDeviceContext context,
                                      const OptixProgramGroupDesc *programDescriptions,
                                      unsigned int numProgramGroups,
                                      const OptixProgramGroupOptions *options,
                                      char *logString,
                                      size_t *logStringSize,
                                      OptixProgramGroup *programGroups)
  {
    return g_optixFunctionTable.optixProgramGroupCreate(context, programDescriptions, numProgramGroups, options,
                                                        logString, logStringSize, programGroups);
  }

  OptixResult optixProgramGroupDestroy(OptixProgramGroup programGroup)
  {
    return g_optixFunctionTable.optixProgramGroupDestroy(programGroup);
  }

  OptixResult optixProgramGroupGetStackSize(OptixProgramGroup programGroup, OptixStackSizes *stackSizes)
  {
    return g_optixFunctionTable.optixProgramGroupGetStackSize(programGroup, stackSizes);
  }

  OptixResult optixPipelineCreate(OptixDeviceContext context,
                                  const OptixPipelineCompileOptions *pipelineCompileOptions,
                                  const OptixPipelineLinkOptions *pipelineLinkOptions,
                                  const OptixProgramGroup *programGroups,
                                  unsigned int numProgramGroups,
                                  char *logString,
                                  size_t *logStringSize,
                                  OptixPipeline *pipeline)
  {
    return g_optixFunctionTable.optixPipelineCreate(context, pipelineCompileOptions, pipelineLinkOptions, programGroups,
                                                    numProgramGroups, logString, logStringSize, pipeline);
  }

  OptixResult optixPipelineDestroy(OptixPipeline pipeline)
  {
    return g_optixFunctionTable.optixPipelineDestroy(pipeline);
  }

  OptixResult optixPipelineSetStackSize(OptixPipeline pipeline,
                                        unsigned int directCallableStackSizeFromTraversal,
                                        unsigned int directCallableStackSizeFromState,
                                        unsigned int continuationStackSize,
                                        unsigned int maxTraversableGraphDepth)
  {
    return g_optixFunctionTable.optixPipelineSetStackSize(pipeline, directCallableStackSizeFromTraversal, directCallableStackSizeFromState,
                                                          continuationStackSize, maxTraversableGraphDepth);
  }

  OptixResult optixAccelComputeMemoryUsage(OptixDeviceContext context,
                                           const OptixAccelBuildOptions *accelOptions,
                                           const OptixBuildInput *buildInputs,
                                           unsigned int numBuildInputs,
                                           OptixAccelBufferSizes *bufferSizes)
  {
    return g_optixFunctionTable.optixAccelComputeMemoryUsage(context, accelOptions, buildInputs, numBuildInputs, bufferSizes);
  }

  OptixResult optixAccelBuild(OptixDeviceContext context,
                              CUstream stream,
                              const OptixAccelBuildOptions *accelOptions,
                              const OptixBuildInput *buildInputs,
                              unsigned int numBuildInputs,
                              CUdeviceptr tempBuffer,
                              size_t tempBufferSizeInBytes,
                              CUdeviceptr outputBuffer,
                              size_t outputBufferSizeInBytes,
                              OptixTraversableHandle *outputHandle,
                              const OptixAccelEmitDesc *emittedProperties,
                              unsigned int numEmittedProperties)
  {
    return g_optixFunctionTable.optixAccelBuild(context, stream, accelOptions, buildInputs, numBuildInputs, tempBuffer,
                                                tempBufferSizeInBytes, outputBuffer, outputBufferSizeInBytes,
                                                outputHandle, emittedProperties, numEmittedProperties);
  }

  OptixResult optixAccelGetRelocationInfo(OptixDeviceContext context, OptixTraversableHandle handle, OptixAccelRelocationInfo *info)
  {
    return g_optixFunctionTable.optixAccelGetRelocationInfo(context, handle, info);
  }

  OptixResult optixAccelCheckRelocationCompatibility(OptixDeviceContext context, const OptixAccelRelocationInfo *info, int *compatible)
  {
    return g_optixFunctionTable.optixAccelCheckRelocationCompatibility(context, info, compatible);
  }

  OptixResult optixAccelRelocate(OptixDeviceContext context,
                                 CUstream stream,
                                 const OptixAccelRelocationInfo *info,
                                 CUdeviceptr instanceTraversableHandles,
                                 size_t numInstanceTraversableHandles,
                                 CUdeviceptr targetAccel,
                                 size_t targetAccelSizeInBytes,
                                 OptixTraversableHandle *targetHandle)
  {
    return g_optixFunctionTable.optixAccelRelocate(context, stream, info, instanceTraversableHandles, numInstanceTraversableHandles,
                                                   targetAccel, targetAccelSizeInBytes, targetHandle);
  }

  OptixResult optixAccelCompact(OptixDeviceContext context,
                                CUstream stream,
                                OptixTraversableHandle inputHandle,
                                CUdeviceptr outputBuffer,
                                size_t outputBufferSizeInBytes,
                                OptixTraversableHandle *outputHandle)
  {
    return g_optixFunctionTable.optixAccelCompact(context, stream, inputHandle, outputBuffer, outputBufferSizeInBytes, outputHandle);
  }

  OptixResult optixConvertPointerToTraversableHandle(OptixDeviceContext onDevice,
                                                     CUdeviceptr pointer,
                                                     OptixTraversableType traversableType,
                                                     OptixTraversableHandle *traversableHandle)
  {
    return g_optixFunctionTable.optixConvertPointerToTraversableHandle(onDevice, pointer, traversableType, traversableHandle);
  }

  OptixResult optixSbtRecordPackHeader(OptixProgramGroup programGroup, void *sbtRecordHeaderHostPointer)
  {
    return g_optixFunctionTable.optixSbtRecordPackHeader(programGroup, sbtRecordHeaderHostPointer);
  }

  OptixResult optixLaunch(OptixPipeline pipeline,
                          CUstream stream,
                          CUdeviceptr pipelineParams,
                          size_t pipelineParamsSize,
                          const OptixShaderBindingTable *sbt,
                          unsigned int width,
                          unsigned int height,
                          unsigned int depth)
  {
    return g_optixFunctionTable.optixLaunch(pipeline, stream, pipelineParams, pipelineParamsSize, sbt, width, height, depth);
  }

  OptixResult optixDenoiserCreate(OptixDeviceContext context, OptixDenoiserModelKind modelKind, const OptixDenoiserOptions *options, OptixDenoiser *returnHandle)
  {
    return g_optixFunctionTable.optixDenoiserCreate(context, modelKind, options, returnHandle);
  }

  OptixResult optixDenoiserCreateWithUserModel(OptixDeviceContext context, const void *data, size_t dataSizeInBytes, OptixDenoiser *returnHandle)
  {
    return g_optixFunctionTable.optixDenoiserCreateWithUserModel(context, data, dataSizeInBytes, returnHandle);
  }

  OptixResult optixDenoiserDestroy(OptixDenoiser handle)
  {
    return g_optixFunctionTable.optixDenoiserDestroy(handle);
  }

  OptixResult optixDenoiserComputeMemoryResources(const OptixDenoiser handle,
                                                  unsigned int maximumInputWidth,
                                                  unsigned int maximumInputHeight,
                                                  OptixDenoiserSizes *returnSizes)
  {
    return g_optixFunctionTable.optixDenoiserComputeMemoryResources(handle, maximumInputWidth, maximumInputHeight, returnSizes);
  }

  OptixResult optixDenoiserSetup(OptixDenoiser denoiser,
                                 CUstream stream,
                                 unsigned int inputWidth,
                                 unsigned int inputHeight,
                                 CUdeviceptr denoiserState,
                                 size_t denoiserStateSizeInBytes,
                                 CUdeviceptr scratch,
                                 size_t scratchSizeInBytes)
  {
    return g_optixFunctionTable.optixDenoiserSetup(denoiser, stream, inputWidth, inputHeight, denoiserState,
                                                   denoiserStateSizeInBytes, scratch, scratchSizeInBytes);
  }

  OptixResult optixDenoiserInvoke(OptixDenoiser handle,
                                  CUstream stream,
                                  const OptixDenoiserParams *params,
                                  CUdeviceptr denoiserData,
                                  size_t denoiserDataSize,
                                  const OptixDenoiserGuideLayer *guideLayer,
                                  const OptixDenoiserLayer *layers,
                                  unsigned int numLayers,
                                  unsigned int inputOffsetX,
                                  unsigned int inputOffsetY,
                                  CUdeviceptr scratch,
                                  size_t scratchSizeInBytes)
  {
    return g_optixFunctionTable.optixDenoiserInvoke(handle, stream, params, denoiserData, denoiserDataSize,
                                                    guideLayer, layers, numLayers,
                                                    inputOffsetX, inputOffsetY, scratch, scratchSizeInBytes);
  }

  OptixResult optixDenoiserComputeIntensity(OptixDenoiser handle,
                                            CUstream stream,
                                            const OptixImage2D *inputImage,
                                            CUdeviceptr outputIntensity,
                                            CUdeviceptr scratch,
                                            size_t scratchSizeInBytes)
  {
    return g_optixFunctionTable.optixDenoiserComputeIntensity(handle, stream, inputImage, outputIntensity, scratch, scratchSizeInBytes);
  }

  OptixResult optixDenoiserComputeAverageColor(OptixDenoiser handle,
                                               CUstream stream,
                                               const OptixImage2D *inputImage,
                                               CUdeviceptr outputAverageColor,
                                               CUdeviceptr scratch,
                                               size_t scratchSizeInBytes)
  {
    return g_optixFunctionTable.optixDenoiserComputeAverageColor(handle, stream, inputImage, outputAverageColor, scratch, scratchSizeInBytes);
  }

#ifdef __cplusplus
}
#endif
