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

//particle data structure using a Structure of Arrays (SoA) design

#ifndef PARTICLE_GPU_H_
#define PARTICLE_GPU_H_

#include <cuda_runtime_api.h>
#include <cuda.h>

#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/sequence.h>

#include "types.h"

struct particle_gpu {
	//state
	float2_t *pos = 0;				//position
	float2_t *vel = 0;				//velocity

	float_t *h   = 0;				//smoothing length
	float_t *rho = 0;				//density
	float_t *p   = 0;				//pressure

	float4_t *S  = 0;				//stress [Sxx, Sxy, Syy, Szz]
	float4_t *R  = 0;				//artificial Stress [Rxx, Rxy, Ryy, -]

	//contact and wear
	float2_t *fc = 0;				//contact force
	float2_t *ft = 0;				//tangential force
	float2_t *n  = 0;				//normal at contacting interface

	//boundary conditions (1 if fixed in SPACE, zero else)
	float_t *fixed = 0;				//particle is fixed in space
	float_t *blanked = 0;			//particle is deactivated
	float_t *tool_particle = 0;		//particle is a tool particle: no mechanical solver, thermal solver only. translated with tool velocity.

	//(equivalent) plastic strain and strain rate
	float_t *eps_pl;
	float_t *eps_pl_dot;

	//temperature
	float_t  *T  = 0;

	//derivatives in time
	float2_t *pos_t = 0;
	float2_t *vel_t = 0;

	float_t  *rho_t = 0;
	float4_t *S_t   = 0;
	float_t  *T_t   = 0;

	//spatial ders
	float4_t *S_der = 0;
	float4_t *v_der = 0;

	//hashing
	int *idx = 0;		//unsigned int should be avoided on gpu
	int *hash = 0;		//		see best practices guide

	//count on host (!)
	unsigned int N;

	//constructors with various initital conditions
	particle_gpu(unsigned int N);
	particle_gpu(float2_t *pos, float2_t *vel_init, float_t *rho, float_t *h, unsigned int N);
	particle_gpu(float2_t *pos, float2_t *vel_init, float_t *rho, float_t *h, float_t *fixed, unsigned int N);
	particle_gpu(float2_t *pos, float2_t *vel_init, float_t *rho, float_t *T_init, float_t *h, float_t *fixed, unsigned int N);
	particle_gpu(float2_t *pos, float2_t *vel_init, float_t *rho, float_t *T_init, float_t *h, float_t *fixed, float_t * tool_p, unsigned int N);
	particle_gpu(float2_t *pos, float2_t *vel_init, float_t *rho, float_t *h, float4_t *Sini, unsigned int N);
};

#endif /* PARTICLE_GPU_H_ */
