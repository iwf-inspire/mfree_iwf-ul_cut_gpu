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

#include "leap_frog.h"

#include "grid_gpu_green.h"
struct inistate_struct {
	float2_t *pos_init;
	float2_t *vel_init;
	float4_t *S_init;
	float_t  *rho_init;
	float_t  *T_init;
	float_t  *T_init_tool;
};

__global__ void init(particle_gpu particles, inistate_struct inistate) {
	unsigned int pidx = blockIdx.x * blockDim.x + threadIdx.x;
	if (pidx >= particles.N) return;
	if (particles.blanked[pidx] == 1.) return;

	inistate.pos_init[pidx] = particles.pos[pidx];
	inistate.vel_init[pidx] = particles.vel[pidx];
	inistate.S_init[pidx]   = particles.S[pidx];
	inistate.rho_init[pidx] = particles.rho[pidx];
	inistate.T_init[pidx]   = particles.T[pidx];
}

__global__ void predict(particle_gpu particles, inistate_struct inistate, float_t dt) {
	unsigned int pidx = blockIdx.x * blockDim.x + threadIdx.x;
	if (pidx >= particles.N) return;
	if (particles.blanked[pidx] == 1.) return;

	particles.pos[pidx].x = inistate.pos_init[pidx].x + 0.5*dt*particles.pos_t[pidx].x;
	particles.pos[pidx].y = inistate.pos_init[pidx].y + 0.5*dt*particles.pos_t[pidx].y;

	particles.vel[pidx].x = inistate.vel_init[pidx].x + 0.5*dt*particles.vel_t[pidx].x;
	particles.vel[pidx].y = inistate.vel_init[pidx].y + 0.5*dt*particles.vel_t[pidx].y;

#ifdef TVF
	particles.vel_adv[pidx].x = inistate.vel_init[pidx].x + 0.5*dt*particles.vel_adv_t[pidx].x;
	particles.vel_adv[pidx].y = inistate.vel_init[pidx].y + 0.5*dt*particles.vel_adv_t[pidx].y;
#endif

	particles.S[pidx].x   = inistate.S_init[pidx].x + 0.5*dt*particles.S_t[pidx].x;
	particles.S[pidx].y   = inistate.S_init[pidx].y + 0.5*dt*particles.S_t[pidx].y;
	particles.S[pidx].z   = inistate.S_init[pidx].z + 0.5*dt*particles.S_t[pidx].z;
	particles.S[pidx].w   = inistate.S_init[pidx].w + 0.5*dt*particles.S_t[pidx].w;

	particles.rho[pidx]   = inistate.rho_init[pidx] + 0.5*dt*particles.rho_t[pidx];

	particles.T[pidx]     = inistate.T_init[pidx]   + 0.5*dt*particles.T_t[pidx];
}

__global__ void correct(particle_gpu particles, inistate_struct inistate, float_t dt) {
	unsigned int pidx = blockIdx.x * blockDim.x + threadIdx.x;
	if (pidx >= particles.N) return;
	if (particles.blanked[pidx] == 1.) return;

	particles.pos[pidx].x = inistate.pos_init[pidx].x + dt*particles.pos_t[pidx].x;
	particles.pos[pidx].y = inistate.pos_init[pidx].y + dt*particles.pos_t[pidx].y;

	particles.vel[pidx].x = inistate.vel_init[pidx].x + dt*particles.vel_t[pidx].x;
	particles.vel[pidx].y = inistate.vel_init[pidx].y + dt*particles.vel_t[pidx].y;

#ifdef TVF
	particles.vel_adv[pidx].x = inistate.vel_init[pidx].x + dt*particles.vel_adv_t[pidx].x;
	particles.vel_adv[pidx].y = inistate.vel_init[pidx].y + dt*particles.vel_adv_t[pidx].y;
#endif

	particles.S[pidx].x   = inistate.S_init[pidx].x + dt*particles.S_t[pidx].x;
	particles.S[pidx].y   = inistate.S_init[pidx].y + dt*particles.S_t[pidx].y;
	particles.S[pidx].z   = inistate.S_init[pidx].z + dt*particles.S_t[pidx].z;
	particles.S[pidx].w   = inistate.S_init[pidx].w + dt*particles.S_t[pidx].w;

	particles.rho[pidx]   = inistate.rho_init[pidx] + dt*particles.rho_t[pidx];

	particles.T[pidx]     = inistate.T_init[pidx]   + dt*particles.T_t[pidx];
}

void leap_frog::step(particle_gpu *particles, grid_base *g) {
	dim3 dG((particles->N + BLOCK_SIZE-1) / BLOCK_SIZE);
	dim3 dB(BLOCK_SIZE);

	inistate_struct inistate;
	inistate.pos_init = pos_init;
	inistate.vel_init = vel_init;
	inistate.S_init   = S_init;
	inistate.rho_init = rho_init;
	inistate.T_init   = T_init;

	//spatial sorting
	g->update_geometry(particles, global_tool, 2.);
	g->assign_hashes(particles, global_tool);
	g->sort(particles, global_tool);
	g->get_cells(particles, cell_start, cell_end);

	//leap frog predictor
	init<<<dG,dB>>>(*particles, inistate);
	predict<<<dG,dB>>>(*particles, inistate, global_dt);

	material_eos(particles);

	corrector_artificial_stress(particles);

	interactions_setup_geometry_constants(g);
	interactions_monaghan(particles, cell_start, cell_end, g->num_cell());

#ifdef Thermal_Conduction_PSE
	interactions_heat_pse(particles, cell_start, cell_end, g->num_cell());
#endif

	material_stress_rate_jaumann(particles);

	contmech_continuity(particles);
	contmech_momentum(particles);
	contmech_advection(particles);

	//leap frog predictor
	correct<<<dG,dB>>>(*particles, inistate, global_dt);

	if (global_tool) {
		material_fric_heat_gen(particles, global_tool->get_vel());
	}

	//plastic state by radial return
	plasticity_johnson_cook(particles);

	//establish contact by penalty method if tool is present
	if (global_tool) {
		tool_gpu_update_tool(global_tool, particles);
		compute_contact_forces(particles);
	}

	//boundary conditions
	perform_boundary_conditions(particles);
	perform_boundary_conditions_thermal(particles);

	//debugging methods
//	debug_check_valid(particles);
	debug_invalidate(particles);
}

leap_frog::leap_frog(unsigned int num_part, unsigned int num_cell) {
	cudaMalloc((void **) &pos_init, sizeof(float2_t)*num_part);
	cudaMalloc((void **) &vel_init, sizeof(float2_t)*num_part);
	cudaMalloc((void **) &S_init,   sizeof(float4_t)*num_part);
	cudaMalloc((void **) &rho_init,   sizeof(float_t)*num_part);
	cudaMalloc((void **) &T_init,   sizeof(float_t)*num_part);

	cudaMalloc((void **) &cell_start, sizeof(int)*num_cell);
	cudaMalloc((void **) &cell_end,   sizeof(int)*num_cell);
}
