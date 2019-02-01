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

//spatial hashing according to green 2008
//	this file contains heavily modified code by nvidia cuda sample "particles"
//	Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

#ifndef GRID_GPU_GREEN_H_
#define GRID_GPU_GREEN_H_

#include "particle_gpu.h"

#include <cuda_runtime_api.h>
#include <cuda.h>

#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/sequence.h>

#include "grid.h"
#include "tool.h"
#include "types.h"

class grid_gpu_green : public grid_base {
public:

	//buffering
	struct device_buffer {
		float2_t *pos        = 0;
		float2_t *vel        = 0;
#ifdef TVF
		float2_t *vel_adv        = 0;
#endif
		float2_t *fc         = 0;
		float2_t *ft         = 0;
		float_t  *h          = 0;
		float_t  *rho        = 0;
		float4_t *S          = 0;
		float_t  *eps_pl     = 0;
		float_t  *eps_pl_dot = 0;
		float_t  *T          = 0;
		float_t  *fixed      = 0;
		float_t  *blanked    = 0;
		float_t  *tool_particle    = 0;

		float2_t *pos_t      = 0;
		float2_t *vel_t      = 0;
#ifdef TVF
		float2_t *vel_adv_t      = 0;
#endif
		float_t  *rho_t      = 0;
		float4_t *S_t        = 0;
		float_t  *T_t        = 0;
	};

private:
	enum buffer_method {copy, swap};

	buffer_method m_buffer_method = buffer_method::swap;
	device_buffer *m_buffer;

	int *m_cell_start = 0;
	int *m_cell_end   = 0;

	int *m_tool_cell_start = 0;
	int *m_tool_cell_end   = 0;

	void alloc_buffer(int num_cell, int num_part);

public:

	void sort(particle_gpu *particles, tool *tool) const override;
	void get_cells(particle_gpu *particles, int *cell_start, int *cell_end) override;

	grid_gpu_green(unsigned int max_cell, unsigned int N);
	grid_gpu_green(int num_part, float2_t bbmin, float2_t bbmax, float_t h);
};

#endif /* GRID_GPU_GREEN_H_ */
