//Copyright ETH Zurich, IWF

//This file is part of mfree_iwf-ul_cut_gpu.

//mfree_iwf is free software: you can redistribute it and/or modify
//it under the terms of the GNU General Public License as published by
//the Free Software Foundation, either version 3 of the License, or
//(at your option) any later version.

//mfree_iwf-ul_cut_gpu is distributed in the hope that it will be useful,
//but WITHOUT ANY WARRANTY; without even the implied warranty of
//MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//GNU General Public License for more details.

//You should have received a copy of the GNU General Public License
//along with mfree_iwf-ul_cut_gpu.  If not, see <http://www.gnu.org/licenses/>.

//set central solver controls
//	single/double precision
//	method for thermal solution (PSE or Brookshaw, see Eldgredge 2002 and Brookshaw 1994)

#ifndef TYPES_H_
#define TYPES_H_

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

#include <glm/glm.hpp>

#define USE_DOUBLE

#ifdef USE_DOUBLE

	// built in vector types
	#define float_t double
	#define float2_t double2
	#define float3_t double3
	#define float4_t double4

	// texture types and texture fetching
	#define float_tex_t  int2
	#define float2_tex_t int4
	#define float4_tex_t int4

	#define make_float2_t make_double2
	#define make_float3_t make_double3
	#define make_float4_t make_double4

	#define texfetch1 fetch_double
	#define texfetch2 fetch_double
	#define texfetch4 fetch_double2

	// glm types
	#define mat2x2_t glm::dmat2x2
	#define mat3x3_t glm::dmat3x3
	#define vec3_t glm::dvec3
	#define vec2_t glm::dvec2

	#define BLOCK_SIZE 256

#else

	// built in vector types
	#define float_t float
	#define float2_t float2
	#define float3_t float3
	#define float4_t float4

	#define make_float2_t make_float2
	#define make_float3_t make_float3
	#define make_float4_t make_float4

	// texture types
	#define float_tex_t  float
	#define float2_tex_t float2
	#define float4_tex_t float4

	#define texfetch1 tex1Dfetch
	#define texfetch2 tex1Dfetch
	#define texfetch4 tex1Dfetch

	// glm types
	#define mat2x2_t glm::mat2x2
	#define mat3x3_t glm::mat3x3
	#define vec3_t   glm::vec3
	#define vec2_t   glm::vec2

	#define BLOCK_SIZE 256

#endif

//chose thermal solver
#define Thermal_Conduction_Brookshaw
//#define Thermal_Conduction_PSE

bool check_cuda_error();
bool check_cuda_error(const char *marker);

#endif /* TYPES_H_ */
