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

  /*! perform 'rotation' of float a by amount b. Both a and b must be
      in 0,1 range, the result will be (a+1) clamped to that same
      range (ie, it is the value a shifted by the amount b to the
      right, and re-entering the [0,1) range on the left if it
      "rotates" out on the right */
  inline __both__ float rotate(const float a, const float b)
  {
    float sum = a+b;
    return (sum-1.f)<0.f?sum:(sum-1.f);
  }

  /*! perform 'rotation' of float a by amount b. Both a and b must be
      in 0,1 range, the result will be (a+1) clamped to that same
      range (ie, it is the value a shifted by the amount b to the
      right, and re-entering the [0,1) range on the left if it
      "rotates" out on the right */
  inline __both__ vec2f rotate(const vec2f a, const vec2f b) 
  { return vec2f(rotate(a.x,b.x),rotate(a.y,b.y)); }
  
} // ::gdt
