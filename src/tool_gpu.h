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

#ifndef TOOL_GPU_H_
#define TOOL_GPU_H_

#include "constants_structs.h"
#include "particle_gpu.h"
#include "tool.h"
#include "types.h"

extern float_t global_dt;

//communcate constants to tool subsystem
void tool_gpu_set_up_tool(tool *tool, float_t alpha, phys_constants phys);

//move tool with specified velocity
void tool_gpu_update_tool(tool *tool, particle_gpu *particles);

//contact and friction force computation
void compute_contact_forces(particle_gpu *particles);

#endif /* TOOL_GPU_H_ */
