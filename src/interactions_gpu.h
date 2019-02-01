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

//this module contains all kernels which perform particles interactions
//	particle interactions are made efficient using cell lists obtained from spatial hashing (c.f. grid_green.* or grid_rothlin.*)
//	this is the heart of both the mechanical and thermal solver

#ifndef INTERACTIONS_GPU_H_
#define INTERACTIONS_GPU_H_

#include "particle_gpu.h"
#include "grid_gpu_green.h"
#include "constants_structs.h"
#include "tool.h"

#include <stdio.h>

extern int global_step;

//communicate grid information to interaction system
//		ATTN: needs to be called before any of the interaciton methods
void interactions_setup_geometry_constants(grid_base *g);

//performs all interactions needed to compute the spatial derivatives according to monaghan, gray 2001 including
//XSPH, artificial viscosity and artificial stresses.
//		NOTE: if Thermal_Conduction_Brookshaw is defined the laplacian of the thermal field is computed in addition
void interactions_monaghan(particle_gpu *particles, const int *cell_start, const int *cell_end, int num_cell);

//performs Particle Strenght Exchange (PSE) to compute the laplacian of the thermal field
//		NOTE: by only called if Thermal_Conduction_PSE is defined
void interactions_heat_pse(particle_gpu *particles, const int *cell_start, const int *cell_end, int num_cell);

//set up simulation constants
void interactions_setup_physical_constants(phys_constants phys);
void interactions_setup_corrector_constants(corr_constants corr);
void interactions_setup_thermal_constants_workpiece(trml_constants trml);
void interactions_setup_thermal_constants_tool(trml_constants trml, tool *tool);

#endif /* INTERACTIONS_GPU_H_ */
