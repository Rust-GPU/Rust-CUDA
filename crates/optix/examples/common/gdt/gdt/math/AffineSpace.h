// ======================================================================== //
// Copyright 2018-2019 Ingo Wald                                            //
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

/* originally taken (and adapted) from ospray, under following license */

// ======================================================================== //
// Copyright 2009-2018 Intel Corporation                                    //
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

#include "LinearSpace.h"
#include "box.h"

namespace gdt {

#define VectorT typename L::vector_t
#define ScalarT typename L::vector_t::scalar_t

  ////////////////////////////////////////////////////////////////////////////////
  // Affine Space
  ////////////////////////////////////////////////////////////////////////////////

  template<typename L>
  struct GDT_INTERFACE AffineSpaceT
  {
    L l;           /*< linear part of affine space */
    VectorT p;     /*< affine part of affine space */

    ////////////////////////////////////////////////////////////////////////////////
    // Constructors, Assignment, Cast, Copy Operations
    ////////////////////////////////////////////////////////////////////////////////

    inline AffineSpaceT           ( ) = default;
    inline AffineSpaceT           ( const AffineSpaceT& other ) { l = other.l; p = other.p; }
    inline AffineSpaceT           ( const L           & other ) { l = other  ; p = VectorT(zero); }
    inline AffineSpaceT& operator=( const AffineSpaceT& other ) { l = other.l; p = other.p; return *this; }

    inline AffineSpaceT( const VectorT& vx, const VectorT& vy, const VectorT& vz, const VectorT& p ) : l(vx,vy,vz), p(p) {}
    inline AffineSpaceT( const L& l, const VectorT& p ) : l(l), p(p) {}

    template<typename L1> inline AffineSpaceT( const AffineSpaceT<L1>& s ) : l(s.l), p(s.p) {}

    ////////////////////////////////////////////////////////////////////////////////
    // Constants
    ////////////////////////////////////////////////////////////////////////////////

    inline AffineSpaceT( ZeroTy ) : l(zero), p(zero) {}
    inline AffineSpaceT( OneTy )  : l(one),  p(zero) {}

    /*! return matrix for scaling */
    static inline AffineSpaceT scale(const VectorT& s) { return L::scale(s); }

    /*! return matrix for translation */
    static inline AffineSpaceT translate(const VectorT& p) { return AffineSpaceT(one,p); }

    /*! return matrix for rotation, only in 2D */
    static inline AffineSpaceT rotate(const ScalarT& r) { return L::rotate(r); }

    /*! return matrix for rotation around arbitrary point (2D) or axis (3D) */
    static inline AffineSpaceT rotate(const VectorT& u, const ScalarT& r) { return L::rotate(u,r); }

    /*! return matrix for rotation around arbitrary axis and point, only in 3D */
    static inline AffineSpaceT rotate(const VectorT& p, const VectorT& u, const ScalarT& r) { return translate(+p) * rotate(u,r) * translate(-p);  }

    /*! return matrix for looking at given point, only in 3D; right-handed coordinate system */
    static inline AffineSpaceT lookat(const VectorT& eye, const VectorT& point, const VectorT& up) {
      VectorT Z = normalize(point-eye);
      VectorT U = normalize(cross(Z,up));
      VectorT V = cross(U,Z);
      return AffineSpaceT(L(U,V,Z),eye);
    }

  };

  ////////////////////////////////////////////////////////////////////////////////
  // Unary Operators
  ////////////////////////////////////////////////////////////////////////////////

  template<typename L> inline AffineSpaceT<L> operator -( const AffineSpaceT<L>& a ) { return AffineSpaceT<L>(-a.l,-a.p); }
  template<typename L> inline AffineSpaceT<L> operator +( const AffineSpaceT<L>& a ) { return AffineSpaceT<L>(+a.l,+a.p); }
  template<typename L> inline AffineSpaceT<L>        rcp( const AffineSpaceT<L>& a ) { L il = rcp(a.l); return AffineSpaceT<L>(il,-(il*a.p)); }

  ////////////////////////////////////////////////////////////////////////////////
  // Binary Operators
  ////////////////////////////////////////////////////////////////////////////////

  template<typename L> inline AffineSpaceT<L> operator +( const AffineSpaceT<L>& a, const AffineSpaceT<L>& b ) { return AffineSpaceT<L>(a.l+b.l,a.p+b.p); }
  template<typename L> inline AffineSpaceT<L> operator -( const AffineSpaceT<L>& a, const AffineSpaceT<L>& b ) { return AffineSpaceT<L>(a.l-b.l,a.p-b.p); }

  template<typename L> inline AffineSpaceT<L> operator *( const ScalarT        & a, const AffineSpaceT<L>& b ) { return AffineSpaceT<L>(a*b.l,a*b.p); }
  template<typename L> inline AffineSpaceT<L> operator *( const AffineSpaceT<L>& a, const AffineSpaceT<L>& b ) { return AffineSpaceT<L>(a.l*b.l,a.l*b.p+a.p); }
  template<typename L> inline AffineSpaceT<L> operator /( const AffineSpaceT<L>& a, const AffineSpaceT<L>& b ) { return a * rcp(b); }
  template<typename L> inline AffineSpaceT<L> operator /( const AffineSpaceT<L>& a, const ScalarT        & b ) { return a * rcp(b); }

  template<typename L> inline AffineSpaceT<L>& operator *=( AffineSpaceT<L>& a, const AffineSpaceT<L>& b ) { return a = a * b; }
  template<typename L> inline AffineSpaceT<L>& operator *=( AffineSpaceT<L>& a, const ScalarT        & b ) { return a = a * b; }
  template<typename L> inline AffineSpaceT<L>& operator /=( AffineSpaceT<L>& a, const AffineSpaceT<L>& b ) { return a = a / b; }
  template<typename L> inline AffineSpaceT<L>& operator /=( AffineSpaceT<L>& a, const ScalarT        & b ) { return a = a / b; }

  template<typename L> inline __both__ const VectorT xfmPoint (const AffineSpaceT<L>& m, const VectorT& p) { return madd(VectorT(p.x),m.l.vx,madd(VectorT(p.y),m.l.vy,madd(VectorT(p.z),m.l.vz,m.p))); }
  template<typename L> inline __both__ const VectorT xfmVector(const AffineSpaceT<L>& m, const VectorT& v) { return xfmVector(m.l,v); }
  template<typename L> inline __both__ const VectorT xfmNormal(const AffineSpaceT<L>& m, const VectorT& n) { return xfmNormal(m.l,n); }


  // template<typename S, bool A=false>
  // inline const box_t<S,3,A> 
  // xfmBounds(const AffineSpaceT<LinearSpace3<vec_t<S,3,A>>> &m, 
  //           const box_t<S,3,A> &b)
  // {
  //   box_t<S,3,A> dst = empty;
  //   const vec_t<S,3,A> p0(b.lower.x,b.lower.y,b.lower.z); dst.extend(xfmPoint(m,p0));
  //   const vec_t<S,3,A> p1(b.lower.x,b.lower.y,b.upper.z); dst.extend(xfmPoint(m,p1));
  //   const vec_t<S,3,A> p2(b.lower.x,b.upper.y,b.lower.z); dst.extend(xfmPoint(m,p2));
  //   const vec_t<S,3,A> p3(b.lower.x,b.upper.y,b.upper.z); dst.extend(xfmPoint(m,p3));
  //   const vec_t<S,3,A> p4(b.upper.x,b.lower.y,b.lower.z); dst.extend(xfmPoint(m,p4));
  //   const vec_t<S,3,A> p5(b.upper.x,b.lower.y,b.upper.z); dst.extend(xfmPoint(m,p5));
  //   const vec_t<S,3,A> p6(b.upper.x,b.upper.y,b.lower.z); dst.extend(xfmPoint(m,p6));
  //   const vec_t<S,3,A> p7(b.upper.x,b.upper.y,b.upper.z); dst.extend(xfmPoint(m,p7));
  //   return dst;
  // }

  ////////////////////////////////////////////////////////////////////////////////
  /// Comparison Operators
  ////////////////////////////////////////////////////////////////////////////////

  template<typename L> inline bool operator ==( const AffineSpaceT<L>& a, const AffineSpaceT<L>& b ) { return a.l == b.l && a.p == b.p; }
  template<typename L> inline bool operator !=( const AffineSpaceT<L>& a, const AffineSpaceT<L>& b ) { return a.l != b.l || a.p != b.p; }

  ////////////////////////////////////////////////////////////////////////////////
  // Output Operators
  ////////////////////////////////////////////////////////////////////////////////

  template<typename L> inline std::ostream& operator<<(std::ostream& cout, const AffineSpaceT<L>& m) {
    return cout << "{ l = " << m.l << ", p = " << m.p << " }";
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Type Aliases
  ////////////////////////////////////////////////////////////////////////////////

  using AffineSpace2f      = AffineSpaceT<LinearSpace2f>;
  using AffineSpace3f      = AffineSpaceT<LinearSpace3f>;
  using AffineSpace3fa     = AffineSpaceT<LinearSpace3fa>;
  using OrthonormalSpace3f = AffineSpaceT<Quaternion3f >;

  using affine2f = AffineSpace2f;
  using affine3f = AffineSpace3f;

  ////////////////////////////////////////////////////////////////////////////////
  /*! Template Specialization for 2D: return matrix for rotation around point (rotation around arbitrarty vector is not meaningful in 2D) */
  template<> inline AffineSpace2f AffineSpace2f::rotate(const vec2f& p, const float& r)
  { return translate(+p) * AffineSpace2f(LinearSpace2f::rotate(r)) * translate(-p); }

#undef VectorT
#undef ScalarT

} // ::gdt
