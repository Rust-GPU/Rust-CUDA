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

#include <type_traits>

namespace gdt {

  // inline __both__ float min(float x, float y) { return fminf(x,y); }
  // inline __both__ float max(float x, float y) { return fmaxf(x,y); }
  
  // =======================================================
  // scalar functors
  // =======================================================

  template<typename T>
  inline __both__ T divRoundUp(const T &a, const T &b)
  { //causes issues on ubuntu16-gcc: static_assert(std::numeric_limits<T>::is_integer);
    return T((a+b-1)/b); }
  
  // =======================================================
  // vector specializations of those scalar functors
  // =======================================================

  template<typename T, int N> inline __both__
  bool any_less_than(const vec_t<T,N> &a, const vec_t<T,N> &b)
  { for (int i=0;i<N;i++) if (a[i] < b[i]) return true; return false; }
  
  template<typename T, int N> inline __both__
  bool any_greater_than(const vec_t<T,N> &a, const vec_t<T,N> &b)
  { for (int i=0;i<N;i++) if (a[i] > b[i]) return true; return false; }

  // -------------------------------------------------------
  // unary functors
  // -------------------------------------------------------

  template<typename T>
  inline __both__ T clamp(const T &val, const T &lo, const T &hi)
  { return min(hi,max(lo,val)); }
  
  template<typename T>
  inline __both__ T clamp(const T &val, const T &hi)
  { return clamp(val,(T)0,hi); }
  
#define _define_float_functor(func)                                     \
  template<typename T> inline __both__ vec_t<T,2> func(const vec_t<T,2> &v) \
  { return vec_t<T,2>(func(v.x),func(v.y)); }                           \
                                                                        \
  template<typename T> inline __both__ vec_t<T,3> func(const vec_t<T,3> &v) \
  { return vec_t<T,3>(func(v.x),func(v.y),func(v.z)); }                 \
                                                                        \
  template<typename T> inline __both__ vec_t<T,4> func(const vec_t<T,4> &v) \
  { return vec_t<T,4>(func(v.x),func(v.y),func(v.z),func(v.w)); }       \

  _define_float_functor(rcp)
  _define_float_functor(sin)
  _define_float_functor(cos)
  _define_float_functor(abs)
  _define_float_functor(saturate)
  
#undef _define_float_functor
  
  // -------------------------------------------------------
  // binary functors
  // -------------------------------------------------------
  // template<typename T>
  // __both__ vec_t<T,2> divRoundUp(const vec_t<T,2> &a, const vec_t<T,2> &b)
  // { return vec_t<T,2>(divRoundUp(a.x,b.x),divRoundUp(a.y,b.y)); }
  
  // template<typename T>
  // __both__ vec_t<T,3> divRoundUp(const vec_t<T,3> &a, const vec_t<T,3> &b)
  // { return vec_t<T,3>(divRoundUp(a.x,b.x),divRoundUp(a.y,b.y),divRoundUp(a.z,b.z)); }

#define _define_binary_functor(fct)                                     \
  template<typename T>                                                  \
  __both__ vec_t<T,1> fct(const vec_t<T,1> &a, const vec_t<T,1> &b)     \
  {                                                                     \
    return vec_t<T,1>(fct(a.x,b.x));                                    \
  }                                                                     \
                                                                        \
  template<typename T>                                                  \
  __both__ vec_t<T,2> fct(const vec_t<T,2> &a, const vec_t<T,2> &b)     \
  {                                                                     \
    return vec_t<T,2>(fct(a.x,b.x),                                     \
                      fct(a.y,b.y));                                    \
  }                                                                     \
                                                                        \
  template<typename T>                                                  \
  __both__ vec_t<T,3> fct(const vec_t<T,3> &a, const vec_t<T,3> &b)     \
  {                                                                     \
    return vec_t<T,3>(fct(a.x,b.x),                                     \
                      fct(a.y,b.y),                                     \
                      fct(a.z,b.z));                                    \
  }                                                                     \
                                                                        \
  template<typename T1, typename T2>                                    \
  __both__ vec_t<typename BinaryOpResultType<T1,T2>::type,3>            \
  fct(const vec_t<T1,3> &a, const vec_t<T2,3> &b)                       \
  {                                                                     \
    return vec_t<typename BinaryOpResultType<T1,T2>::type,3>            \
      (fct(a.x,b.x),                                                    \
       fct(a.y,b.y),                                                    \
       fct(a.z,b.z));                                                   \
  }                                                                     \
                                                                        \
  template<typename T>                                                  \
  __both__ vec_t<T,4> fct(const vec_t<T,4> &a, const vec_t<T,4> &b)     \
  {                                                                     \
    return vec_t<T,4>(fct(a.x,b.x),                                     \
                      fct(a.y,b.y),                                     \
                      fct(a.z,b.z),                                     \
                      fct(a.w,b.w));                                    \
  }                                                                     \

  
  _define_binary_functor(divRoundUp)
  _define_binary_functor(min)
  _define_binary_functor(max)
#undef _define_binary_functor




  

  // -------------------------------------------------------
  // binary operators
  // -------------------------------------------------------
#define _define_operator(op)                                            \
  /* vec op vec */                                                      \
  template<typename T>                                                  \
  inline __both__ vec_t<T,2> operator op(const vec_t<T,2> &a,           \
                                         const vec_t<T,2> &b)           \
  { return vec_t<T,2>(a.x op b.x, a.y op b.y); }                        \
                                                                        \
  template<typename T>                                                  \
  inline __both__ vec_t<T,3> operator op(const vec_t<T,3> &a,           \
                                         const vec_t<T,3> &b)           \
  { return vec_t<T,3>(a.x op b.x, a.y op b.y, a.z op b.z); }            \
                                                                        \
  template<typename T>                                                  \
  inline __both__ vec_t<T,4> operator op(const vec_t<T,4> &a,           \
                                         const vec_t<T,4> &b)           \
  { return vec_t<T,4>(a.x op b.x,a.y op b.y,a.z op b.z,a.w op b.w); }   \
                                                                        \
  /* vec op scalar */                                                   \
  template<typename T>                                                  \
  inline __both__ vec_t<T,2> operator op(const vec_t<T,2> &a,           \
                                         const T &b)                    \
  { return vec_t<T,2>(a.x op b, a.y op b); }                            \
                                                                        \
  template<typename T1, typename T2>                                    \
  inline __both__ vec_t<typename BinaryOpResultType<T1,T2>::type,3>     \
  operator op(const vec_t<T1,3> &a, const T2 &b)                        \
  { return vec_t<typename BinaryOpResultType<T1,T2>::type,3>            \
      (a.x op b, a.y op b, a.z op b);                                   \
  }                                                                     \
                                                                        \
  template<typename T>                                                  \
  inline __both__ vec_t<T,4> operator op(const vec_t<T,4> &a,           \
                                         const T &b)                    \
  { return vec_t<T,4>(a.x op b, a.y op b, a.z op b, a.w op b); }        \
                                                                        \
  /* scalar op vec */                                                   \
  template<typename T>                                                  \
  inline __both__ vec_t<T,2> operator op(const T &a,                    \
                                         const vec_t<T,2> &b)           \
  { return vec_t<T,2>(a op b.x, a op b.y); }                            \
                                                                        \
  template<typename T>                                                  \
  inline __both__ vec_t<T,3> operator op(const T &a,                    \
                                         const vec_t<T,3> &b)           \
  { return vec_t<T,3>(a op b.x, a op b.y, a op b.z); }                  \
                                                                        \
  template<typename T>                                                  \
  inline __both__ vec_t<T,4> operator op(const T &a,                    \
                                         const vec_t<T,4> &b)           \
  { return vec_t<T,4>(a op b.x, a op b.y, a op b.z, a op b.w); }        \
                                                                        \
                                                                        \
    
  _define_operator(*);
  _define_operator(/);
  _define_operator(+);
  _define_operator(-);
  
#undef _define_operator




  // -------------------------------------------------------
  // unary operators
  // -------------------------------------------------------

  template<typename T>
  inline __both__ vec_t<T,2> operator-(const vec_t<T,2> &v)
  { return vec_t<T,2>(-v.x, -v.y); }
  
  template<typename T>
  inline __both__ vec_t<T,2> operator+(const vec_t<T,2> &v)
  { return vec_t<T,2>(v.x, v.y); }

  template<typename T>
  inline __both__ vec_t<T,3> operator-(const vec_t<T,3> &v)
  { return vec_t<T,3>(-v.x, -v.y, -v.z); }
  
  template<typename T>
  inline __both__ vec_t<T,3> operator+(const vec_t<T,3> &v)
  { return vec_t<T,3>(v.x, v.y, v.z); }



  // -------------------------------------------------------
  // binary op-assign operators
  // -------------------------------------------------------
#define  _define_op_assign_operator(operator_op,op)                     \
  /* vec op vec */                                                      \
  template<typename T, typename OT>                                     \
  inline __both__ vec_t<T,2> &operator_op(vec_t<T,2> &a,                \
                                          const vec_t<OT,2> &b)         \
  {                                                                     \
    a.x op (T)b.x;                                                      \
    a.y op (T)b.y;                                                      \
    return a;                                                           \
  }                                                                     \
                                                                        \
  template<typename T, typename OT>                                     \
  inline __both__ vec_t<T,3> &operator_op(vec_t<T,3> &a,                \
                                          const vec_t<OT,3> &b)         \
  {                                                                     \
    a.x op (T)b.x;                                                      \
    a.y op (T)b.y;                                                      \
    a.z op (T)b.z;                                                      \
    return a;                                                           \
  }                                                                     \
                                                                        \
  template<typename T, typename OT>                                     \
  inline __both__ vec_t<T,4> &operator_op(vec_t<T,4> &a,                \
                                          const vec_t<OT,4> &b)         \
  {                                                                     \
    a.x op (T)b.x;                                                      \
    a.y op (T)b.y;                                                      \
    a.z op (T)b.z;                                                      \
    a.w op (T)b.w;                                                      \
    return a;                                                           \
  }                                                                     \
                                                                        \
  /* vec op scalar */                                                   \
  template<typename T, typename OT>                                     \
  inline __both__ vec_t<T,2> &operator_op(vec_t<T,2> &a,                \
                                          const OT &b)                  \
  { a.x op (T)b; a.y op (T)b; return a; }                               \
                                                                        \
  template<typename T, typename OT>                                     \
  inline __both__ vec_t<T,3> &operator_op(vec_t<T,3> &a,                \
                                          const OT &b)                  \
  { a.x op (T)b; a.y op (T)b; a.z op (T)b; return a; }                  \
                                                                        \
  template<typename T, typename OT>                                     \
  inline __both__ vec_t<T,4> &operator_op(vec_t<T,4> &a,                \
                                          const OT &b)                  \
  { a.x op (T)b; a.y op (T)b; a.z op (T)b; a.w op (T)b; return a; }     \
    
  _define_op_assign_operator(operator*=,*=);
  _define_op_assign_operator(operator/=,/=);
  _define_op_assign_operator(operator+=,+=);
  _define_op_assign_operator(operator-=,-=);
  
#undef _define_op_assign_operator


  template<typename T>
  __both__ T reduce_min(const vec_t<T,1> &v) { return v.x; }
  template<typename T>
  __both__ T reduce_min(const vec_t<T,2> &v) { return min(v.x,v.y); }
  template<typename T>
  __both__ T reduce_min(const vec_t<T,3> &v) { return min(min(v.x,v.y),v.z); }
  template<typename T>
  __both__ T reduce_min(const vec_t<T,4> &v) { return min(min(v.x,v.y),min(v.z,v.w)); }
  template<typename T>
  __both__ T reduce_max(const vec_t<T,2> &v) { return max(v.x,v.y); }
  template<typename T>
  __both__ T reduce_max(const vec_t<T,3> &v) { return max(max(v.x,v.y),v.z); }
  template<typename T>
  __both__ T reduce_max(const vec_t<T,4> &v) { return max(max(v.x,v.y),max(v.z,v.w)); }


  template<typename T, int N>
  __both__ vec_t<T,3> madd(const vec_t<T,N> &a, const vec_t<T,N> &b, const vec_t<T,N> &c)
  {
    return a*b + c;
  }


  template<typename T, int N>
  __both__ int arg_max(const vec_t<T,N> &v)
  {
    int biggestDim = 0;
    for (int i=1;i<N;i++)
      if (abs(v[i]) > abs(v[biggestDim])) biggestDim = i;
    return biggestDim;
  }
  

  // less, for std::set, std::map, etc
  template<typename T>
  __both__ bool operator<(const vec_t<T,3> &a, const vec_t<T,3> &b)
  {
    if (a.x < b.x) return true;
    if (a.x == b.x && a.y < b.y) return true;
    if (a.x == b.x && a.y == b.y && a.z < b.z) return true;
    return false;
    // return
    //   (a.x < b.x) |
    //   ((a.x == b.x) & ((a.y < b.y) |
    //                    ((a.y == b.y) & (a.z < b.z))));
  }

  /*! helper function that creates a semi-random color from an ID */
  inline __both__ vec3f randomColor(int i)
  {
    int r = unsigned(i)*13*17 + 0x234235;
    int g = unsigned(i)*7*3*5 + 0x773477;
    int b = unsigned(i)*11*19 + 0x223766;
    return vec3f((r&255)/255.f,
                 (g&255)/255.f,
                 (b&255)/255.f);
  }

  /*! helper function that creates a semi-random color from an ID */
  inline __both__ vec3f randomColor(size_t idx)
  {
    unsigned int r = (unsigned int)(idx*13*17 + 0x234235);
    unsigned int g = (unsigned int)(idx*7*3*5 + 0x773477);
    unsigned int b = (unsigned int)(idx*11*19 + 0x223766);
    return vec3f((r&255)/255.f,
                 (g&255)/255.f,
                 (b&255)/255.f);
  }

  /*! helper function that creates a semi-random color from an ID */
  template<typename T>
  inline __both__ vec3f randomColor(const T *ptr)
  {
    return randomColor((size_t)ptr);
  }

  
} // ::gdt
