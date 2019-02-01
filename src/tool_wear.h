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

//module containing methods to compute the accumulated Usui wear rate (Usui 84)
//	variables are copied from GPU, wear is then computed on CPU

#ifndef TOOL_WEAR_H_
#define TOOL_WEAR_H_

#include <stdio.h>

#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/extrema.h>

#include "constants_structs.h"
#include "particle_gpu.h"
#include "types.h"

class tool_wear {
private:
	float_t m_K = 0.;
	float_t m_alpha = 0.;
	phys_constants m_physical_constants;
	vec2_t m_tool_vel;

	float2_t *m_vel_buf;	//these are on host. decided to copy back for evaluating usui
	float2_t *m_fc_buf;	    //	(just like data is copied back for writing result files)
	float2_t *m_ft_buf;
	float_t  *m_h_buf;
	float2_t *m_normal_buf;
	float_t  *m_T_buf;

	float_t m_accum_wear = 0;

public:
	//get accumulated wear
	float_t get_accum_wear() const;
	//eval wear _rates_
	void eval_usui(const particle_gpu *particles, float_t dt, float_t &usui_min, float_t &usui_max, float_t &usui_avg);
	tool_wear(float_t usui_K, float_t usui_alpha, unsigned int N_part, phys_constants physical_constants, vec2_t tool_vel);
};

#endif /* TOOL_WEAR_H_ */
