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

//this module contains the complete code to setup metal cutting simulations and some preliminary benchmarks

#ifndef BENCHMARKS_H_
#define BENCHMARKS_H_

#include <thrust/device_vector.h>
#include "constants_structs.h"
#include "particle_gpu.h"
#include "types.h"
#include "actions_gpu.h"
#include "interactions_gpu.h"
#include "tool.h"
#include "tool_gpu.h"
#include "types.h"
#include "debug.h"
#include "grid.h"
#include "tool_wear.h"
#include "grid_gpu_rothlin.h"

extern tool *global_tool;
extern tool_wear *global_wear;

//rubber ring impact, see for example gray & monaghan 2001 (verifies elastic stage)
//		this example uses SI units
particle_gpu *setup_rings(int nbox, grid_base **grid);
//plastic - plastic wall impact. see for example rothlin 2019 (verifies plastic stage)
//		this example uses SI units
particle_gpu *setup_impact(int nbox, grid_base **grid);
//reference cut as defined by ruttimann 2012
//		this example uses bomb units!!!! [musec, kg, cm]
particle_gpu *setup_ref_cut(int ny, grid_base **grid, float_t rake = 0., float_t clear = 0., float_t chamfer = 0., float_t speed = 0., float_t feed = 0.01);

#endif /* BENCHMARKS_H_ */
