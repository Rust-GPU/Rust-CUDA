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

#include "vec.h"

namespace gdt
{
  ////////////////////////////////////////////////////////////////
  // Quaternion Struct
  ////////////////////////////////////////////////////////////////

  template<typename T>
  struct QuaternionT
  {
    typedef vec_t<T,3> Vector;

    ////////////////////////////////////////////////////////////////////////////////
    /// Construction
    ////////////////////////////////////////////////////////////////////////////////

    __both__ QuaternionT           ( void )                     { }
    __both__ QuaternionT           ( const QuaternionT& other ) { r = other.r; i = other.i; j = other.j; k = other.k; }
    __both__ QuaternionT& operator=( const QuaternionT& other ) { r = other.r; i = other.i; j = other.j; k = other.k; return *this; }

    __both__          QuaternionT( const T& r       ) : r(r), i(zero), j(zero), k(zero) {}
    __both__ explicit QuaternionT( const Vector& v ) : r(zero), i(v.x), j(v.y), k(v.z) {}
    __both__          QuaternionT( const T& r, const T& i, const T& j, const T& k ) : r(r), i(i), j(j), k(k) {}
    __both__          QuaternionT( const T& r, const Vector& v ) : r(r), i(v.x), j(v.y), k(v.z) {}

    __inline QuaternionT( const Vector& vx, const Vector& vy, const Vector& vz );
    __inline QuaternionT( const T& yaw, const T& pitch, const T& roll );

    ////////////////////////////////////////////////////////////////////////////////
    /// Constants
    ////////////////////////////////////////////////////////////////////////////////

    __both__ QuaternionT( ZeroTy ) : r(zero), i(zero), j(zero), k(zero) {}
    __both__ QuaternionT( OneTy  ) : r( one), i(zero), j(zero), k(zero) {}

    /*! return quaternion for rotation around arbitrary axis */
    static __both__ QuaternionT rotate(const Vector& u, const T& r) {
      return QuaternionT<T>(cos(T(0.5)*r),sin(T(0.5)*r)*normalize(u));
    }

    /*! returns the rotation axis of the quaternion as a vector */
    __both__ const Vector v( ) const { return Vector(i, j, k); }

  public:
    T r, i, j, k;
  };

  template<typename T> __both__ QuaternionT<T> operator *( const T             & a, const QuaternionT<T>& b ) { return QuaternionT<T>(a * b.r, a * b.i, a * b.j, a * b.k); }
  template<typename T> __both__ QuaternionT<T> operator *( const QuaternionT<T>& a, const T             & b ) { return QuaternionT<T>(a.r * b, a.i * b, a.j * b, a.k * b); }

  ////////////////////////////////////////////////////////////////
  // Unary Operators
  ////////////////////////////////////////////////////////////////

  template<typename T> __both__ QuaternionT<T> operator +( const QuaternionT<T>& a ) { return QuaternionT<T>(+a.r, +a.i, +a.j, +a.k); }
  template<typename T> __both__ QuaternionT<T> operator -( const QuaternionT<T>& a ) { return QuaternionT<T>(-a.r, -a.i, -a.j, -a.k); }
  template<typename T> __both__ QuaternionT<T> conj      ( const QuaternionT<T>& a ) { return QuaternionT<T>(a.r, -a.i, -a.j, -a.k); }
  template<typename T> __both__ T              abs       ( const QuaternionT<T>& a ) { return sqrt(a.r*a.r + a.i*a.i + a.j*a.j + a.k*a.k); }
  template<typename T> __both__ QuaternionT<T> rcp       ( const QuaternionT<T>& a ) { return conj(a)*rcp(a.r*a.r + a.i*a.i + a.j*a.j + a.k*a.k); }
  template<typename T> __both__ QuaternionT<T> normalize ( const QuaternionT<T>& a ) { return a*rsqrt(a.r*a.r + a.i*a.i + a.j*a.j + a.k*a.k); }

  ////////////////////////////////////////////////////////////////
  // Binary Operators
  ////////////////////////////////////////////////////////////////

  template<typename T> __both__ QuaternionT<T> operator +( const T             & a, const QuaternionT<T>& b ) { return QuaternionT<T>(a + b.r,  b.i,  b.j,  b.k); }
  template<typename T> __both__ QuaternionT<T> operator +( const QuaternionT<T>& a, const T             & b ) { return QuaternionT<T>(a.r + b, a.i, a.j, a.k); }
  template<typename T> __both__ QuaternionT<T> operator +( const QuaternionT<T>& a, const QuaternionT<T>& b ) { return QuaternionT<T>(a.r + b.r, a.i + b.i, a.j + b.j, a.k + b.k); }
  template<typename T> __both__ QuaternionT<T> operator -( const T             & a, const QuaternionT<T>& b ) { return QuaternionT<T>(a - b.r, -b.i, -b.j, -b.k); }
  template<typename T> __both__ QuaternionT<T> operator -( const QuaternionT<T>& a, const T             & b ) { return QuaternionT<T>(a.r - b, a.i, a.j, a.k); }
  template<typename T> __both__ QuaternionT<T> operator -( const QuaternionT<T>& a, const QuaternionT<T>& b ) { return QuaternionT<T>(a.r - b.r, a.i - b.i, a.j - b.j, a.k - b.k); }

  template<typename T> __both__ typename QuaternionT<T>::Vector       operator *( const QuaternionT<T>& a, const typename QuaternionT<T>::Vector      & b ) { return (a*QuaternionT<T>(b)*conj(a)).v(); }
  template<typename T> __both__ QuaternionT<T> operator *( const QuaternionT<T>& a, const QuaternionT<T>& b ) {
    return QuaternionT<T>(a.r*b.r - a.i*b.i - a.j*b.j - a.k*b.k,
                          a.r*b.i + a.i*b.r + a.j*b.k - a.k*b.j,
                          a.r*b.j - a.i*b.k + a.j*b.r + a.k*b.i,
                          a.r*b.k + a.i*b.j - a.j*b.i + a.k*b.r);
  }
  template<typename T> __both__ QuaternionT<T> operator /( const T             & a, const QuaternionT<T>& b ) { return a*rcp(b); }
  template<typename T> __both__ QuaternionT<T> operator /( const QuaternionT<T>& a, const T             & b ) { return a*rcp(b); }
  template<typename T> __both__ QuaternionT<T> operator /( const QuaternionT<T>& a, const QuaternionT<T>& b ) { return a*rcp(b); }

  template<typename T> __both__ QuaternionT<T>& operator +=( QuaternionT<T>& a, const T             & b ) { return a = a+b; }
  template<typename T> __both__ QuaternionT<T>& operator +=( QuaternionT<T>& a, const QuaternionT<T>& b ) { return a = a+b; }
  template<typename T> __both__ QuaternionT<T>& operator -=( QuaternionT<T>& a, const T             & b ) { return a = a-b; }
  template<typename T> __both__ QuaternionT<T>& operator -=( QuaternionT<T>& a, const QuaternionT<T>& b ) { return a = a-b; }
  template<typename T> __both__ QuaternionT<T>& operator *=( QuaternionT<T>& a, const T             & b ) { return a = a*b; }
  template<typename T> __both__ QuaternionT<T>& operator *=( QuaternionT<T>& a, const QuaternionT<T>& b ) { return a = a*b; }
  template<typename T> __both__ QuaternionT<T>& operator /=( QuaternionT<T>& a, const T             & b ) { return a = a*rcp(b); }
  template<typename T> __both__ QuaternionT<T>& operator /=( QuaternionT<T>& a, const QuaternionT<T>& b ) { return a = a*rcp(b); }

  template<typename T> __both__ typename QuaternionT<T>::Vector 
  xfmPoint ( const QuaternionT<T>& a, 
             const typename QuaternionT<T>::Vector&       b ) 
  { return (a*QuaternionT<T>(b)*conj(a)).v(); }

  template<typename T> __both__ typename QuaternionT<T>::Vector 
  xfmQuaternion( const QuaternionT<T>& a, 
                 const typename QuaternionT<T>::Vector&       b ) 
  { return (a*QuaternionT<T>(b)*conj(a)).v(); }


  template<typename T> __both__ typename QuaternionT<T>::Vector 
  xfmNormal( const QuaternionT<T>& a, 
             const typename QuaternionT<T>::Vector&       b ) 
  { return (a*QuaternionT<T>(b)*conj(a)).v(); }

  ////////////////////////////////////////////////////////////////////////////////
  /// Comparison Operators
  ////////////////////////////////////////////////////////////////////////////////

  template<typename T> __both__ bool operator ==( const QuaternionT<T>& a, const QuaternionT<T>& b ) { return a.r == b.r && a.i == b.i && a.j == b.j && a.k == b.k; }

  template<typename T> __both__ bool operator !=( const QuaternionT<T>& a, const QuaternionT<T>& b ) { return a.r != b.r || a.i != b.i || a.j != b.j || a.k != b.k; }


  ////////////////////////////////////////////////////////////////////////////////
  /// Orientation Functions
  ////////////////////////////////////////////////////////////////////////////////

  template<typename T> 
  QuaternionT<T>::QuaternionT(const typename QuaternionT<T>::Vector& vx, 
                              const typename QuaternionT<T>::Vector& vy, 
                              const typename QuaternionT<T>::Vector& vz )
  {
    if ( vx.x + vy.y + vz.z >= T(zero) )
    {
      const T t = T(one) + (vx.x + vy.y + vz.z);
      const T s = rsqrt(t)*T(0.5f);
      r = t*s;
      i = (vy.z - vz.y)*s;
      j = (vz.x - vx.z)*s;
      k = (vx.y - vy.x)*s;
    }
    else if ( vx.x >= max(vy.y, vz.z) )
    {
      const T t = (T(one) + vx.x) - (vy.y + vz.z);
      const T s = rsqrt(t)*T(0.5f);
      r = (vy.z - vz.y)*s;
      i = t*s;
      j = (vx.y + vy.x)*s;
      k = (vz.x + vx.z)*s;
    }
    else if ( vy.y >= vz.z ) // if ( vy.y >= max(vz.z, vx.x) )
    {
      const T t = (T(one) + vy.y) - (vz.z + vx.x);
      const T s = rsqrt(t)*T(0.5f);
      r = (vz.x - vx.z)*s;
      i = (vx.y + vy.x)*s;
      j = t*s;
      k = (vy.z + vz.y)*s;
    }
    else //if ( vz.z >= max(vy.y, vx.x) )
    {
      const T t = (T(one) + vz.z) - (vx.x + vy.y);
      const T s = rsqrt(t)*T(0.5f);
      r = (vx.y - vy.x)*s;
      i = (vz.x + vx.z)*s;
      j = (vy.z + vz.y)*s;
      k = t*s;
    }
  }

  template<typename T> QuaternionT<T>::QuaternionT( const T& yaw, const T& pitch, const T& roll )
  {
    const T cya = cos(yaw  *T(0.5f));
    const T cpi = cos(pitch*T(0.5f));
    const T cro = cos(roll *T(0.5f));
    const T sya = sin(yaw  *T(0.5f));
    const T spi = sin(pitch*T(0.5f));
    const T sro = sin(roll *T(0.5f));
    r = cro*cya*cpi + sro*sya*spi;
    i = cro*cya*spi + sro*sya*cpi;
    j = cro*sya*cpi - sro*cya*spi;
    k = sro*cya*cpi - cro*sya*spi;
  }

  ////////////////////////////////////////////////////////////////////////////////
  /// Output Operators
  ////////////////////////////////////////////////////////////////////////////////

  template<typename T> static std::ostream& operator<<(std::ostream& cout, const QuaternionT<T>& q) {
    return cout << "{ r = " << q.r << ", i = " << q.i << ", j = " << q.j << ", k = " << q.k << " }";
  }

  /*! default template instantiations */
  typedef QuaternionT<float>  Quaternion3f;
  typedef QuaternionT<double> Quaternion3d;
}
