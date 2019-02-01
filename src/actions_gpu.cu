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

#include "actions_gpu.h"

#include "plasticity.cuh"

#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

static bool m_plastic = false;
static bool m_thermal = false;			//consider thermal conduction in workpiece
static bool m_fric_heat_gen = false;	//consider that friction produces heat

__constant__ static phys_constants physics;
__constant__ static corr_constants correctors;
__constant__ static joco_constants johnson_cook;
__constant__ static trml_constants thermals_wp;
__constant__ static trml_constants thermals_tool;

__device__ __forceinline__ float_t stress_angle(float_t sxx, float_t sxy, float_t syy, float_t eps) {
	float_t numer = 2.*sxy;
	float_t denom = sxx - syy + eps;
	return 0.5*atan2f(numer,denom);
}

__global__ void do_material_eos(const float_t *__restrict__ rho, float_t *__restrict__ p, const float_t *__restrict__ in_tool, int N) {
	unsigned int pidx = blockIdx.x * blockDim.x + threadIdx.x;
	if (pidx >= N) return;
	if (in_tool[pidx]) return;

	float_t rho0 = physics.rho0;
	float_t c0   = sqrtf(physics.K/rho0);
	float_t rhoi = rho[pidx];
	p[pidx] = c0*c0*(rhoi - rho0);

}

__global__ void do_corrector_artificial_stress(const float_t *__restrict__ rho, const float_t *__restrict__ p, const float4_t *__restrict__ S, const float_t *__restrict__ in_tool,
		float4_t *__restrict__ R, int N) {
	unsigned int pidx = blockIdx.x * blockDim.x + threadIdx.x;
	if (pidx >= N) return;
	if (in_tool[pidx]) return;

	float_t eps  = correctors.stresseps;

	float_t rhoi = rho[pidx];
	float_t pi   = p[pidx];
	float4_t Si  = S[pidx];

	float_t sxx  = Si.x;
	float_t sxy  = Si.y;
	float_t syy  = Si.z;

	sxx -= pi;
	syy -= pi;

	float_t rhoi21 = 1./(rhoi*rhoi);

	float_t theta = stress_angle(sxx,sxy,syy,0.);

	float_t cos_theta = cosf(theta);
	float_t sin_theta = sinf(theta);

	float_t cos_theta2 = cos_theta*cos_theta;
	float_t sin_theta2 = sin_theta*sin_theta;

	float_t rot_sxx = cos_theta2*sxx + 2.0*cos_theta*sin_theta*sxy + sin_theta2*syy;
	float_t rot_syy = sin_theta2*sxx - 2.0*cos_theta*sin_theta*sxy + cos_theta2*syy;

	float_t rot_rxx = 0.;
	float_t rot_ryy = 0.;

	if (rot_sxx > 0) rot_rxx = -eps*rot_sxx*rhoi21;
	if (rot_syy > 0) rot_ryy = -eps*rot_syy*rhoi21;

	float4_t Ri = make_float4_t(cos_theta2*rot_rxx + sin_theta2*rot_ryy,
			cos_theta*sin_theta*(rot_rxx - rot_ryy),
			sin_theta2*rot_rxx + cos_theta2*rot_ryy,
			0.);

	R[pidx] = Ri;
}

__global__ void do_material_stress_rate_jaumann(const float4_t *__restrict__ v_der, const float4_t *__restrict__ Stress, const float_t *in_tool,
		float4_t *S_t, int N) {
	unsigned int pidx = blockIdx.x * blockDim.x + threadIdx.x;
	if (pidx >= N) return;
	if (in_tool[pidx] == 1.) return;

	float_t G = physics.G;

	float4_t vi_der = v_der[pidx];
	float4_t Si     = Stress[pidx];

	float_t vx_x = vi_der.x;
	float_t vx_y = vi_der.y;
	float_t vy_x = vi_der.z;
	float_t vy_y = vi_der.w;

	float_t Sxx = Si.x;
	float_t Sxy = Si.y;
	float_t Syy = Si.z;
	float_t Szz = Si.w;

	mat3x3_t epsdot = mat3x3_t(vx_x, 0.5*(vx_y + vy_x), 0., 0.5*(vx_y + vy_x), vy_y, 0., 0., 0., 0.);
	mat3x3_t omega  = mat3x3_t(0.  , 0.5*(vy_x - vx_y), 0., 0.5*(vx_y - vy_x), 0., 0., 0., 0., 0.);
	mat3x3_t S      = mat3x3_t(Sxx, Sxy, 0., Sxy, Syy, 0., 0., 0., Szz);
	mat3x3_t I      = mat3x3_t(1.);

	float_t trace_epsdot = epsdot[0][0] + epsdot[1][1] + epsdot[2][2];

	mat3x3_t Si_t = float_t(2.)*G*(epsdot - float_t(1./3.)*trace_epsdot*I) + omega*S + S*glm::transpose(omega);	//Belytschko (3.7.9)

	S_t[pidx].x = Si_t[0][0];
	S_t[pidx].y = Si_t[0][1];
	S_t[pidx].z = Si_t[1][1];
	S_t[pidx].w = Si_t[2][2];
}

__global__ void do_material_fric_heat_gen(const float2_t * __restrict__ vel, const float2_t * __restrict__ f_fric, const float2_t * __restrict__ n, const float_t *in_tool,
		float_t *__restrict__ T, float2_t vel_tool, float_t dt, int N) {
	unsigned int pidx = blockIdx.x * blockDim.x + threadIdx.x;
	if (pidx >= N) return;
	if (in_tool[pidx]) return;

	const float_t eta = thermals_wp.eta;

	//compute F_fric_mag;
	float2_t f_T =  f_fric[pidx];
	float_t  f_fric_mag = sqrtf(f_T.x*f_T.x + f_T.y*f_T.y);

	if (f_fric_mag == 0.) {
		return;
	}

	//compute v_rel
	float2_t normal     = n[pidx];
	float2_t v_particle = vel[pidx];
	float2_t v_diff     = make_float2_t(v_particle.x-vel_tool.x, v_particle.y-vel_tool.y);

	float_t  v_diff_dot_normal = v_diff.x*normal.x + v_diff.y*normal.y;
	float2_t v_relative = make_float2_t(v_diff.x -  v_diff_dot_normal, v_diff.y - v_diff_dot_normal);

	float_t  v_rel_mag  = sqrtf(v_relative.x*v_relative.x + v_relative.y*v_relative.y);

	T[pidx] += eta*dt*f_fric_mag*v_rel_mag/(thermals_wp.cp*physics.mass);
}

__global__ void do_contmech_continuity(const float_t *__restrict__ rho, const float4_t *__restrict__ v_der, const float_t *in_tool,
		float_t *__restrict__ rho_t, int N) {
	unsigned int pidx = blockIdx.x * blockDim.x + threadIdx.x;
	if (pidx >= N) return;
	if (in_tool[pidx]) return;

	double rhoi  = rho[pidx];
	float4_t vi_der = v_der[pidx];

	float_t vx_x = vi_der.x;
	float_t vy_y = vi_der.w;

	rho_t[pidx] = -rhoi*(vx_x + vy_y);
}

__global__ void do_contmech_momentum(const float4_t *__restrict__ S_der, const float2_t *__restrict__ fc, const float2_t *__restrict__ ft, const float_t *in_tool,
		float2_t *__restrict__ vel_t, int N) {
	unsigned int pidx = blockIdx.x * blockDim.x + threadIdx.x;
	if (pidx >= N) return;
	if (in_tool[pidx]) return;

	float_t mass = physics.mass;

	float4_t Si_der = S_der[pidx];
	float2_t fci    = fc[pidx];
	float2_t fti    = ft[pidx];
	float2_t veli_t = vel_t[pidx];

	float_t Sxx_x = Si_der.x;
	float_t Sxy_y = Si_der.y;
	float_t Sxy_x = Si_der.z;
	float_t Syy_y = Si_der.w;

	float_t fcx   = fci.x;
	float_t fcy   = fci.y;

	float_t ftx   = fti.x;
	float_t fty   = fti.y;

	//redundant mult and div by rho elimnated in der stress
	veli_t.x += Sxx_x + Sxy_y + fcx / mass + ftx / mass;
	veli_t.y += Sxy_x + Syy_y + fcy / mass + fty / mass;

	vel_t[pidx] = veli_t;
}

__global__ void do_contmech_advection(const float2_t *__restrict__ vel, const float_t *in_tool,
		float2_t *__restrict__ pos_t, int N) {
	unsigned int pidx = blockIdx.x * blockDim.x + threadIdx.x;
	if (pidx >= N) return;
	if (in_tool[pidx]) return;

	float2_t veli   = vel[pidx];
	float2_t posi_t = pos_t[pidx];

	float2_t posi_t_new;
	posi_t_new.x = posi_t.x + veli.x;
	posi_t_new.y = posi_t.y + veli.y;

	pos_t[pidx] = posi_t_new;
}

__global__ void do_plasticity_johnson_cook(particle_gpu particles, float_t dt) {
	unsigned int pidx = blockIdx.x * blockDim.x + threadIdx.x;
	if (pidx >= particles.N) return;
	if (particles.tool_particle[pidx]) return;

	float_t mu = physics.G;

	float4_t S = particles.S[pidx];
	float_t Strialxx = S.x;
	float_t Strialyy = S.z;
	float_t Strialzz = S.w;
	float_t Strialxy = S.y;

	float_t norm_Strial = sqrtf(Strialxx*Strialxx + Strialyy*Strialyy + Strialzz*Strialzz + Strialxy*Strialxy);

	float_t p = particles.p[pidx];

	float_t cxx = Strialxx - p;
	float_t cyy = Strialyy - p;
	float_t czz = Strialzz - p;
	float_t cxy = Strialxy;

	float_t eps_pl     = particles.eps_pl[pidx];
	float_t eps_pl_dot = particles.eps_pl_dot[pidx];
	float_t T          = particles.T[pidx];

	float_t svm = sqrtf((cxx*cxx + cyy*cyy + czz*czz) - cxx * cyy - cxx * czz - cyy * czz + 3.0 * cxy * cxy);
	float_t sigma_Y = sigma_yield(johnson_cook, eps_pl, eps_pl_dot, T);

	// elastic case
	if (svm < sigma_Y) {
		particles.eps_pl_dot[pidx] = 0.;
		return;
	}

	float_t delta_lambda = solve_secant(johnson_cook, fmax(eps_pl_dot*dt*sqrt(2./3.), 1e-8), 1e-6,
			norm_Strial, eps_pl, T, dt, physics.G);

	float_t eps_pl_new = eps_pl + sqrtf(2.0/3.0) * fmaxf(delta_lambda,0.);
	float_t eps_pl_dot_new = sqrtf(2.0/3.0) *  fmaxf(delta_lambda,0.) / dt;

	particles.eps_pl[pidx] = eps_pl_new;
	particles.eps_pl_dot[pidx] = eps_pl_dot_new;

	float4_t S_new;
	S_new.x = Strialxx - Strialxx/norm_Strial*delta_lambda*2.*mu;
	S_new.z = Strialyy - Strialyy/norm_Strial*delta_lambda*2.*mu;
	S_new.w = Strialzz - Strialzz/norm_Strial*delta_lambda*2.*mu;
	S_new.y = Strialxy - Strialxy/norm_Strial*delta_lambda*2.*mu;

	particles.S[pidx] = S_new;

	//plastic work to heat
	if (thermals_wp.tq != 0.) {
		float_t delta_eps_pl = eps_pl_new - eps_pl;
		float_t sigma_Y = sigma_yield(johnson_cook, eps_pl_new, eps_pl_dot_new, T);
		float_t rho = particles.rho[pidx];
		particles.T[pidx] += thermals_wp.tq/(thermals_wp.cp*rho)*delta_eps_pl*sigma_Y;
	}
}

__global__ void do_boundary_conditions_thermal(particle_gpu particles) {
	unsigned int pidx = blockIdx.x * blockDim.x + threadIdx.x;
	if (pidx >= particles.N) return;

	if (particles.fixed[pidx] == 1.) {
		particles.T[pidx] = thermals_wp.T_init;
	}
}

__global__ void do_boundary_conditions(particle_gpu particles) {
	unsigned int pidx = blockIdx.x * blockDim.x + threadIdx.x;
	if (pidx >= particles.N) return;
	if (particles.tool_particle[pidx]) return;

	if (particles.fixed[pidx]) {
		particles.vel[pidx].x = 0.;
		particles.vel[pidx].y = 0.;
		particles.fc[pidx].x = 0.;
		particles.fc[pidx].y = 0.;
		particles.pos_t[pidx].x = 0.;
		particles.pos_t[pidx].y = 0.;
		particles.vel_t[pidx].x = 0.;
		particles.vel_t[pidx].y = 0.;
	}
}

__device__ __forceinline__ bool isnaninf(float_t val) {
	return isnan(val) || isinf(val);
}

__global__ void do_invalidate(particle_gpu particles) {
	unsigned int pidx = blockIdx.x * blockDim.x + threadIdx.x;
	if (pidx >= particles.N) return;

	if (particles.blanked[pidx] == 1.) {
		return;
	}

	bool invalid = false;

	invalid = invalid || isnaninf(particles.pos_t[pidx].x);
	invalid = invalid || isnaninf(particles.pos_t[pidx].y);

	invalid = invalid || isnaninf(particles.vel_t[pidx].x);
	invalid = invalid || isnaninf(particles.vel_t[pidx].y);

	invalid = invalid || isnaninf(particles.S_t[pidx].x);
	invalid = invalid || isnaninf(particles.S_t[pidx].y);
	invalid = invalid || isnaninf(particles.S_t[pidx].z);
	invalid = invalid || isnaninf(particles.S_t[pidx].w);

	invalid = invalid || isnaninf(particles.rho_t[pidx]);
	invalid = invalid || isnaninf(particles.T_t[pidx]);

	if (invalid) {
		particles.blanked[pidx] = 1.;
		printf("invalidated particle %d due to nan at %f %f, %f %f, %f %f %f %f %f %f\n",
				pidx, particles.pos[pidx].x, particles.pos[pidx].y,
				particles.pos_t[pidx].x, particles.pos_t[pidx].y,
				particles.vel_t[pidx].x, particles.vel_t[pidx].x,
				particles.S_t[pidx].x, particles.S_t[pidx].y, particles.S_t[pidx].z, particles.S_t[pidx].w);
	}
}

//---------------------------------------------------------------------

// float2 + struct
struct add_float2 {
    __device__ float2_t operator()(const float2_t& a, const float2_t& b) const {
        float2_t r;
        r.x = a.x + b.x;
        r.y = a.y + b.y;
        return r;
    }
 };

void material_eos(particle_gpu *particles) {
	dim3 dG((particles->N + BLOCK_SIZE-1) / BLOCK_SIZE);
	dim3 dB(BLOCK_SIZE);
	do_material_eos<<<dG,dB>>>(particles->rho, particles->p, particles->tool_particle, particles->N);
}

void corrector_artificial_stress(particle_gpu *particles) {
	dim3 dG((particles->N + BLOCK_SIZE-1) / BLOCK_SIZE);
	dim3 dB(BLOCK_SIZE);
	do_corrector_artificial_stress<<<dG,dB>>>(particles->rho, particles->p, particles->S, particles->tool_particle, particles->R, particles->N);
}

void material_stress_rate_jaumann(particle_gpu *particles) {
	dim3 dG((particles->N + BLOCK_SIZE-1) / BLOCK_SIZE);
	dim3 dB(BLOCK_SIZE);
	do_material_stress_rate_jaumann<<<dG,dB>>>(particles->v_der, particles->S, particles->tool_particle, particles->S_t, particles->N);
}

void contmech_continuity(particle_gpu *particles) {
	dim3 dG((particles->N + BLOCK_SIZE-1) / BLOCK_SIZE);
	dim3 dB(BLOCK_SIZE);
	do_contmech_continuity<<<dG,dB>>>(particles->rho, particles->v_der, particles->tool_particle, particles->rho_t, particles->N);
}

void contmech_momentum(particle_gpu *particles) {
	dim3 dG((particles->N + BLOCK_SIZE-1) / BLOCK_SIZE);
	dim3 dB(BLOCK_SIZE);
	do_contmech_momentum<<<dG,dB>>>(particles->S_der, particles->fc, particles->ft, particles->tool_particle, particles->vel_t, particles->N);
}

void contmech_advection(particle_gpu *particles) {
	dim3 dG((particles->N + BLOCK_SIZE-1) / BLOCK_SIZE);
	dim3 dB(BLOCK_SIZE);
	do_contmech_advection<<<dG,dB>>>(particles->vel, particles->tool_particle, particles->pos_t, particles->N);
}

void plasticity_johnson_cook(particle_gpu *particles) {
	if (!m_plastic) return;
	dim3 dG((particles->N + BLOCK_SIZE-1) / BLOCK_SIZE);
	dim3 dB(BLOCK_SIZE);
	do_plasticity_johnson_cook<<<dG,dB>>>(*particles, global_dt);
}

void perform_boundary_conditions_thermal(particle_gpu *particles) {
	if (!m_thermal) return;
	dim3 dG((particles->N + BLOCK_SIZE-1) / BLOCK_SIZE);
	dim3 dB(BLOCK_SIZE);
	do_boundary_conditions_thermal<<<dG,dB>>>(*particles);
}

void perform_boundary_conditions(particle_gpu *particles) {
	dim3 dG((particles->N + BLOCK_SIZE-1) / BLOCK_SIZE);
	dim3 dB(BLOCK_SIZE);
	do_boundary_conditions<<<dG,dB>>>(*particles);
}

void debug_check_valid(particle_gpu *particles) {
	thrust::device_ptr<float2_t> t_pos(particles->pos);
	float2_t ini;
	ini.x = 0.;
	ini.y = 0.;
	ini = thrust::reduce(t_pos, t_pos + particles->N, ini, add_float2());

	if (isnan(ini.x) || isnan(ini.y)) {
		printf("nan found!\n");
		exit(-1);
	}
}

void debug_invalidate(particle_gpu *particles) {
	dim3 dG((particles->N + BLOCK_SIZE-1) / BLOCK_SIZE);
	dim3 dB(BLOCK_SIZE);
	do_invalidate<<<dG,dB>>>(*particles);
}

void actions_setup_physical_constants(phys_constants physics_h) {
	cudaMemcpyToSymbol(physics, &physics_h, sizeof(phys_constants), 0, cudaMemcpyHostToDevice);
}

void actions_setup_corrector_constants(corr_constants correctors_h) {
	cudaMemcpyToSymbol(correctors, &correctors_h, sizeof(corr_constants), 0, cudaMemcpyHostToDevice);
}

void actions_setup_johnson_cook_constants(joco_constants johnson_cook_h) {
	cudaMemcpyToSymbol(johnson_cook, &johnson_cook_h, sizeof(joco_constants), 0, cudaMemcpyHostToDevice);
	m_plastic = true;
}

void actions_setup_thermal_constants_wp(trml_constants thermal_h) {
	cudaMemcpyToSymbol(thermals_wp, &thermal_h, sizeof(trml_constants), 0, cudaMemcpyHostToDevice);
	if (thermal_h.tq != 0.) {
		printf("considering generation of heat due to plastic work\n");
	}

	if (thermal_h.eta != 0.) {
		printf("considering that friction generates heat\n");
		m_fric_heat_gen = true;
	}
}

void actions_setup_thermal_constants_tool(trml_constants thermal_h) {
	cudaMemcpyToSymbol(thermals_tool, &thermal_h, sizeof(trml_constants), 0, cudaMemcpyHostToDevice);

	if (thermal_h.alpha != 0.) {
		m_thermal = true;
	}
}

void material_fric_heat_gen(particle_gpu *particles, vec2_t vel) {
	if (!m_fric_heat_gen) {
		return;
	}

	dim3 dG((particles->N + BLOCK_SIZE-1) / BLOCK_SIZE);
	dim3 dB(BLOCK_SIZE);
	do_material_fric_heat_gen<<<dG,dB>>>(particles->vel, particles->ft, particles->n,
			particles->tool_particle, particles->T, make_float2_t(vel.x, vel.y), global_dt, particles->N);
}
