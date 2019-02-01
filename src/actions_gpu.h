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

//this module contains all kernels which perform a single loop over the particle array
//	-correctors (artificial stress)
//	-continuity, momentum and advection equation
//	-material modelling (equation of state for pressure, hookes law + jaumann rate for the deviatoric part of the stress)
//	-boundary conditions

#ifndef ACTIONS_GPU_H_
#define ACTIONS_GPU_H_

#include <cuda.h>
#include <cuda_runtime.h>

#include "particle_gpu.h"
#include "constants_structs.h"
#include "types.h"

extern float_t global_dt;

//adiabatic equation of state for pressure term
void material_eos(particle_gpu *particles);
//artificial stresses according to gray & monaghan 2001
void corrector_artificial_stress(particle_gpu *particles);
//stress rate according to jaumann
void material_stress_rate_jaumann(particle_gpu *particles);
//generate frictional heat
void material_fric_heat_gen(particle_gpu *particles, vec2_t vel);

//basic equations
void contmech_continuity(particle_gpu *particles);
void contmech_momentum(particle_gpu *particles);
void contmech_advection(particle_gpu *particles);

//plasticity using the johnson cook model
//		implements the radial return best described in the UINTAH user manual
void plasticity_johnson_cook(particle_gpu *particles);

//boundary conditions
void perform_boundary_conditions_thermal(particle_gpu *particles);
void perform_boundary_conditions(particle_gpu *particles);

//set up simulation constants
void actions_setup_johnson_cook_constants(joco_constants joco);
void actions_setup_physical_constants(phys_constants phys);
void actions_setup_corrector_constants(corr_constants corr);
void actions_setup_thermal_constants_wp(trml_constants thrm);
void actions_setup_thermal_constants_tool(trml_constants thrm);

//debugging (either report or deactivate particles with NaN entries)
void debug_check_valid(particle_gpu *particles);
void debug_invalidate(particle_gpu *particles);

#endif /* ACTIONS_GPU_H_ */
