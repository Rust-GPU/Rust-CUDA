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

namespace gdt {

  // ------------------------------------------------------------------
  // ==
  // ------------------------------------------------------------------

#if __CUDACC__
  template<typename T>
  inline __both__ bool operator==(const vec_t<T,2> &a, const vec_t<T,2> &b)
  { return (a.x==b.x) & (a.y==b.y); }
  
  template<typename T>
  inline __both__ bool operator==(const vec_t<T,3> &a, const vec_t<T,3> &b)
  { return (a.x==b.x) & (a.y==b.y) & (a.z==b.z); }
  
  template<typename T>
  inline __both__ bool operator==(const vec_t<T,4> &a, const vec_t<T,4> &b)
  { return (a.x==b.x) & (a.y==b.y) & (a.z==b.z) & (a.w==b.w); }
#else
  template<typename T>
  inline __both__ bool operator==(const vec_t<T,2> &a, const vec_t<T,2> &b)
  { return a.x==b.x && a.y==b.y; }

  template<typename T>
  inline __both__ bool operator==(const vec_t<T,3> &a, const vec_t<T,3> &b)
  { return a.x==b.x && a.y==b.y && a.z==b.z; }

  template<typename T>
  inline __both__ bool operator==(const vec_t<T,4> &a, const vec_t<T,4> &b)
  { return a.x==b.x && a.y==b.y && a.z==b.z && a.w==b.w; }
#endif
  
  // ------------------------------------------------------------------
  // !=
  // ------------------------------------------------------------------
  
  template<typename T, int N>
  inline __both__ bool operator!=(const vec_t<T,N> &a, const vec_t<T,N> &b)
  { return !(a==b); }
  
} // ::gdt
