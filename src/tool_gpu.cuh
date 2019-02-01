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

//geometrical primitives on the GPU to establish contact with the tool

#ifndef TOOL_CUH_
#define TOOL_CUH_

#include "types.h"
#include "tool.h"

#include <cuda.h>
#include <cuda_runtime.h>

#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

struct line_gpu {
	float_t a;
	float_t b;
	bool vertical;
};

struct segment_gpu {
	float2_t left;
	float2_t right;
	line_gpu l;
	float2_t n;
};

struct circle_segment_gpu {
	float_t  r;
	float_t  t1;
	float_t  t2;
	float2_t   p;
};

#define TOOL_MAX_SEG 5

#endif /* TOOL_CUH_ */
