// ======================================================================== //
// Copyright 2018 Ingo Wald                                                 //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#pragma once

#include <stdio.h>
#include <iostream>
#include <stdexcept>
#include <memory>
#include <assert.h>
#include <string>
#include <math.h>
#ifdef __CUDA_ARCH__
#  include <math_constants.h>
#else
#  include <cmath>
#endif
#include <algorithm>
#ifdef __GNUC__
#  include <sys/time.h>
#  include <stdint.h>
#endif
#include <stdexcept>

#ifdef _WIN32
    #ifndef WIN32_LEAN_AND_MEAN
        #define WIN32_LEAN_AND_MEAN
    #endif
    #include <Windows.h>
    #ifdef min
        #undef min
    #endif
    #ifdef max
        #undef max
    #endif
#endif


#if defined(_MSC_VER)
#  define GDT_DLL_EXPORT __declspec(dllexport)
#  define GDT_DLL_IMPORT __declspec(dllimport)
#elif defined(__clang__) || defined(__GNUC__)
#  define GDT_DLL_EXPORT __attribute__((visibility("default")))
#  define GDT_DLL_IMPORT __attribute__((visibility("default")))
#else
#  define GDT_DLL_EXPORT
#  define GDT_DLL_IMPORT
#endif

#if 1
# define GDT_INTERFACE /* nothing */
#else
//#if defined(GDT_DLL_INTERFACE)
#  ifdef gdt_EXPORTS
#    define GDT_INTERFACE GDT_DLL_EXPORT
#  else
#    define GDT_INTERFACE GDT_DLL_IMPORT
#  endif
//#else
//#  define GDT_INTERFACE /*static lib*/
//#endif
#endif

#ifndef PRINT
# define PRINT(var) std::cout << #var << "=" << var << std::endl;
# define PING std::cout << __FILE__ << "::" << __LINE__ << ": " << __FUNCTION__ << std::endl;
#endif

#if defined(__CUDACC__)
# define __gdt_device   __device__
# define __gdt_host     __host__
#else
# define __gdt_device   /* ignore */
# define __gdt_host     /* ignore */
#endif

# define __both__   __gdt_host __gdt_device


#ifdef __GNUC__
  #define MAYBE_UNUSED __attribute__((unused))
#else
  #define MAYBE_UNUSED
#endif



#define GDT_NOTIMPLEMENTED throw std::runtime_error(std::string(__PRETTY_FUNCTION__)+" not implemented")

#define GDT_TERMINAL_RED "\033[1;31m"
#define GDT_TERMINAL_GREEN "\033[1;32m"
#define GDT_TERMINAL_YELLOW "\033[1;33m"
#define GDT_TERMINAL_BLUE "\033[1;34m"
#define GDT_TERMINAL_RESET "\033[0m"
#define GDT_TERMINAL_DEFAULT GDT_TERMINAL_RESET
#define GDT_TERMINAL_BOLD "\033[1;1m"
     




/*! \namespace gdt GPU Developer Toolbox */
namespace gdt {

#ifdef __CUDACC__
  using ::min;
  using ::max;
  // inline __both__ float abs(float f)      { return fabsf(f); }
  // inline __both__ double abs(double f)    { return fabs(f); }
  using std::abs;
  // inline __both__ float sin(float f) { return ::sinf(f); }
  // inline __both__ double sin(double f) { return ::sin(f); }
  // inline __both__ float cos(float f) { return ::cosf(f); }
  // inline __both__ double cos(double f) { return ::cos(f); }

  using ::saturate;
#else
  using std::min;
  using std::max;
  using std::abs;
  // inline __both__ double sin(double f) { return ::sin(f); }
  inline __both__ float saturate(const float &f) { return min(1.f,max(0.f,f)); }
#endif

  // inline __both__ float abs(float f)      { return fabsf(f); }
  // inline __both__ double abs(double f)    { return fabs(f); }
  inline __both__ float rcp(float f)      { return 1.f/f; }
  inline __both__ double rcp(double d)    { return 1./d; }
  
  inline __both__ int32_t divRoundUp(int32_t a, int32_t b) { return (a+b-1)/b; }
  inline __both__ uint32_t divRoundUp(uint32_t a, uint32_t b) { return (a+b-1)/b; }
  inline __both__ int64_t divRoundUp(int64_t a, int64_t b) { return (a+b-1)/b; }
  inline __both__ uint64_t divRoundUp(uint64_t a, uint64_t b) { return (a+b-1)/b; }
  
#ifdef __CUDACC__
  using ::sin; // this is the double version
  // inline __both__ float sin(float f) { return ::sinf(f); }
  using ::cos; // this is the double version
  // inline __both__ float cos(float f) { return ::cosf(f); }
#else
  using ::sin; // this is the double version
  using ::cos; // this is the double version
#endif

// #ifdef __CUDA_ARCH__
  // using ::sqrt;
  // using ::sqrtf;
// #else
  namespace overloaded {
    /* move all those in a special namespace so they will never get
       included - and thus, conflict with, the default namesapce */
    inline __both__ float  sqrt(const float f)   { return ::sqrtf(f); }
    inline __both__ double sqrt(const double d)   { return ::sqrt(d); }
  }
// #endif
//   inline __both__ float rsqrt(const float f)   { return 1.f/sqrtf(f); }
//   inline __both__ double rsqrt(const double d)   { return 1./sqrt(d); }

#ifdef __WIN32__
#  define osp_snprintf sprintf_s
#else
#  define osp_snprintf snprintf
#endif
  
  /*! added pretty-print function for large numbers, printing 10000000 as "10M" instead */
  inline std::string prettyDouble(const double val) {
    const double absVal = abs(val);
    char result[1000];

    if      (absVal >= 1e+18f) osp_snprintf(result,1000,"%.1f%c",val/1e18f,'E');
    else if (absVal >= 1e+15f) osp_snprintf(result,1000,"%.1f%c",val/1e15f,'P');
    else if (absVal >= 1e+12f) osp_snprintf(result,1000,"%.1f%c",val/1e12f,'T');
    else if (absVal >= 1e+09f) osp_snprintf(result,1000,"%.1f%c",val/1e09f,'G');
    else if (absVal >= 1e+06f) osp_snprintf(result,1000,"%.1f%c",val/1e06f,'M');
    else if (absVal >= 1e+03f) osp_snprintf(result,1000,"%.1f%c",val/1e03f,'k');
    else if (absVal <= 1e-12f) osp_snprintf(result,1000,"%.1f%c",val*1e15f,'f');
    else if (absVal <= 1e-09f) osp_snprintf(result,1000,"%.1f%c",val*1e12f,'p');
    else if (absVal <= 1e-06f) osp_snprintf(result,1000,"%.1f%c",val*1e09f,'n');
    else if (absVal <= 1e-03f) osp_snprintf(result,1000,"%.1f%c",val*1e06f,'u');
    else if (absVal <= 1e-00f) osp_snprintf(result,1000,"%.1f%c",val*1e03f,'m');
    else osp_snprintf(result,1000,"%f",(float)val);

    return result;
  }
  


  inline std::string prettyNumber(const size_t s)
  {
    char buf[1000];
    if (s >= (1024LL*1024LL*1024LL*1024LL)) {
		osp_snprintf(buf, 1000,"%.2fT",s/(1024.f*1024.f*1024.f*1024.f));
    } else if (s >= (1024LL*1024LL*1024LL)) {
		osp_snprintf(buf, 1000, "%.2fG",s/(1024.f*1024.f*1024.f));
    } else if (s >= (1024LL*1024LL)) {
		osp_snprintf(buf, 1000, "%.2fM",s/(1024.f*1024.f));
    } else if (s >= (1024LL)) {
		osp_snprintf(buf, 1000, "%.2fK",s/(1024.f));
    } else {
		osp_snprintf(buf,1000,"%zi",s);
    }
    return buf;
  }
  
  inline double getCurrentTime()
  {
#ifdef _WIN32
    SYSTEMTIME tp; GetSystemTime(&tp);
    return double(tp.wSecond) + double(tp.wMilliseconds) / 1E3;
#else
    struct timeval tp; gettimeofday(&tp,nullptr);
    return double(tp.tv_sec) + double(tp.tv_usec)/1E6;
#endif
  }

  inline bool hasSuffix(const std::string &s, const std::string &suffix)
  {
    return s.substr(s.size()-suffix.size()) == suffix;
  }
}
