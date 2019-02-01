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

//leap frog timestepper

#ifndef LEAP_FROG_H_
#define LEAP_FROG_H_

#include "grid.h"
#include "particle_gpu.h"

#include "actions_gpu.h"
#include "interactions_gpu.h"
#include "tool_gpu.h"

#include "types.h"

extern tool *global_tool;
extern int global_step;
extern float_t global_dt;

class leap_frog{
private:
	float2_t *pos_init = 0;
	float2_t *vel_init = 0;
	float4_t *S_init   = 0;
	float_t  *rho_init = 0;
	float_t  *T_init   = 0;

	int *cell_start = 0;
	int *cell_end   = 0;

public:
	void step(particle_gpu *particles, grid_base *g);
	leap_frog(unsigned int num_part, unsigned int num_cell);
};

#endif /* LEAP_FROG_H_ */
