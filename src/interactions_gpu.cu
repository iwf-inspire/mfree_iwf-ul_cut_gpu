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

#include "interactions_gpu.h"

#include <thrust/device_vector.h>
#include "kernels.cuh"

//physical and other simulation constants in constant device memory
__constant__ static phys_constants physics;
__constant__ static corr_constants correctors;
__constant__ static geom_constants geometry;
__constant__ static trml_constants trml;
__constant__ static trml_constants trml_tool;

//physical constants on host
static trml_constants thermals_workpiece;
static trml_constants thermals_tool;

static bool m_thermal_workpiece = false;
static bool m_thermal_tool = false;

//textures for optimized lookup in interaction kernels
texture<float2_tex_t, 1, cudaReadModeElementType> pos_tex;
texture<float2_tex_t, 1, cudaReadModeElementType> vel_tex;
texture<float4_tex_t, 1, cudaReadModeElementType> S_tex;
texture<float4_tex_t, 1, cudaReadModeElementType> R_tex;
texture<float_tex_t,  1, cudaReadModeElementType> h_tex;
texture<float_tex_t,  1, cudaReadModeElementType> rho_tex;
texture<float_tex_t,  1, cudaReadModeElementType> p_tex;
texture<float_tex_t,  1, cudaReadModeElementType> T_tex;
texture<float_tex_t,  1, cudaReadModeElementType> tool_particle_tex;

texture<int,          1, cudaReadModeElementType> hashes_tex;

texture<float2_tex_t,  1, cudaReadModeElementType> pos_to_tex;
texture<float_tex_t,   1, cudaReadModeElementType> h_to_tex;
texture<float_tex_t,   1, cudaReadModeElementType> T_to_tex;

texture<int,       1, cudaReadModeElementType> hashes_to_tex;

texture<int,      1, cudaReadModeElementType> cells_start_tex;
texture<int,      1, cudaReadModeElementType> cells_end_tex;

#ifdef USE_DOUBLE
static __inline__ __device__ double fetch_double(texture<int2, 1> t, int i) {
	int2 v = tex1Dfetch(t,i);
	return __hiloint2double(v.y, v.x);
}

static __inline__ __device__ double2 fetch_double(texture<int4, 1> t, int i) {
	int4 v = tex1Dfetch(t,i);
	return make_double2(__hiloint2double(v.y, v.x), __hiloint2double(v.w, v.z));
}

static __inline__ __device__ double4 fetch_double2(texture<int4, 1> t, int i) {
	int4 v1 = tex1Dfetch(t, 2*i+0);
	int4 v2 = tex1Dfetch(t, 2*i+1);

	return make_double4(__hiloint2double(v1.y, v1.x), __hiloint2double(v1.w, v1.z),
			__hiloint2double(v2.y, v2.x), __hiloint2double(v2.w, v2.z));
}
#endif

__device__ __forceinline__ void hash(int i, int j, int &idx) {
	idx = i*geometry.ny+j;
}

__device__ __forceinline__ void unhash(int &i, int &j, int idx) {
	i = idx/(geometry.ny);
	j = idx-(i*geometry.ny);
}

__global__ void do_interactions_heat_pse(const float_t *__restrict__ blanked, float_t * T_t, int N, float_t alpha_wp, float_t alpha_tool) {
	unsigned int pidx = blockIdx.x * blockDim.x + threadIdx.x;
	if (pidx >= N) return;
	if (blanked[pidx]) return;

	//load geometrical constants
	int nx = geometry.nx;
	int ny = geometry.ny;

	//load physical constants
	float_t mass  = physics.mass;

	//load particle data at pidx
	float2_t pi = texfetch2(pos_tex, pidx);
	float_t  hi = texfetch1(h_tex, pidx);
	float_t  Ti = texfetch1(T_tex, pidx);

	//unhash and look for neighbor boxes
	int hashi = tex1Dfetch(hashes_tex, pidx);
	int gi,gj;
	unhash(gi, gj, hashi);

	int low_i  = gi-2 < 0 ? 0 : gi-2;
	int low_j  = gj-2 < 0 ? 0 : gj-2;
	int high_i = gi+3 > nx ? nx : gi+3;
	int high_j = gj+3 > ny ? ny : gj+3;

	float_t T_ti = 0.;

	float_t is_tool_particle = texfetch1(tool_particle_tex, pidx);
	float_t alpha = (is_tool_particle == 1.) ? alpha_tool : alpha_wp;

	for (int ii = low_i; ii < high_i; ii++) {
		for (int jj = low_j; jj < high_j; jj++) {
			int idx;
			hash(ii,jj,idx);

			int c_start = tex1Dfetch(cells_start_tex, idx);
			int c_end   = tex1Dfetch(cells_end_tex,   idx);

			if (c_start ==  0xffffffff) continue;

			for (int iter = c_start; iter < c_end; iter++) {

				float2_t pj   = texfetch2(pos_tex, iter);
				float_t  Tj   = texfetch1(T_tex, iter);
				float_t  rhoj = texfetch1(rho_tex, iter);

				float_t w2_pse = lapl_pse(pi, pj, hi);

				T_ti += (Tj-Ti)*w2_pse*mass/rhoj;
			}
		}
	}

	T_t[pidx] = alpha*T_ti;
}

#ifdef Thermal_Conduction_Brookshaw
__global__ void do_interactions_monaghan(const float_t *__restrict__ blanked, float4_t *__restrict__ v_der, float4_t *__restrict__ S_der,
		float2_t *__restrict__ pos_t, float2_t *__restrict__ vel_t, float_t *__restrict__ T_t, unsigned int N) {
#else
	__global__ void do_interactions_monaghan(const float_t *__restrict__ blanked, float4_t *__restrict__ v_der, float4_t *__restrict__ S_der,
			float2_t *__restrict__ pos_t, float2_t *__restrict__ vel_t, unsigned int N) {
#endif
	int pidx = blockIdx.x * blockDim.x + threadIdx.x;

	if (pidx >= N) return;
	if (blanked[pidx] == 1.) return;

	float_t is_tool_particle_i = texfetch1(tool_particle_tex, pidx);

#ifndef Thermal_Conduction_Brookshaw
	if (is_tool_particle_i == 1.) return;
#endif

	//load physical constants
	float_t mass = physics.mass;
	float_t K    = physics.K;
#ifdef Thermal_Conduction_Brookshaw
	float_t thermal_alpha = (is_tool_particle_i == 0.) ? trml.alpha : trml_tool.alpha;
#endif

	//load correction constants
	float_t wdeltap = correctors.wdeltap;
	float_t alpha   = correctors.alpha;
	float_t beta    = correctors.beta;
	float_t eta     = correctors.eta;
	float_t eps     = correctors.xspheps;

	//load geometrical constants
	int nx = geometry.nx;
	int ny = geometry.ny;

	//load particle data at pidx
	float2_t pi   = texfetch2(pos_tex,pidx);
	float2_t vi   = texfetch2(vel_tex,pidx);
	float4_t Si   = texfetch4(S_tex,pidx);
	float4_t Ri   = texfetch4(R_tex,pidx);
	float_t  hi   = texfetch1(h_tex,pidx);
	float_t  rhoi = texfetch1(rho_tex,pidx);
	float_t  prsi = texfetch1(p_tex,pidx);
#ifdef Thermal_Conduction_Brookshaw
	float_t Ti = 0.;
	if (thermal_alpha != 0.) {
		Ti = texfetch1(T_tex,pidx);
	}
#endif

	float_t rhoi21 = 1./(rhoi*rhoi);

	//unhash and look for neighbor boxes
	int hashi = tex1Dfetch(hashes_tex,pidx);
	int gi,gj;
	unhash(gi, gj, hashi);

	int low_i  = gi-2 < 0 ? 0 : gi-2;
	int low_j  = gj-2 < 0 ? 0 : gj-2;
	int high_i = gi+3 > nx ? nx : gi+3;
	int high_j = gj+3 > ny ? ny : gj+3;

	//init vars to be written at pidx
	float4_t vi_der = make_float4_t(0.,0.,0.,0.);
	float4_t Si_der = make_float4_t(0.,0.,0.,0.);
	float2_t vi_t   = make_float2_t(0.,0.);
	float2_t xi_t   = make_float2_t(0.,0.);

#ifdef Thermal_Conduction_Brookshaw
	float_t T_lapl = 0.;
#endif

	for (int ii = low_i; ii < high_i; ii++) {
		for (int jj = low_j; jj < high_j; jj++) {
			int idx;
			hash(ii,jj,idx);

			int c_start = tex1Dfetch(cells_start_tex, idx);
			int c_end   = tex1Dfetch(cells_end_tex,   idx);

			if (c_start ==  0xffffffff) continue;

			for (int iter = c_start; iter < c_end; iter++) {

				if (blanked[iter] == 1.) continue;

				//load vars at neighbor particle
				float2_t pj   = texfetch2(pos_tex,iter);
				float2_t vj   = texfetch2(vel_tex,iter);
				float4_t Sj   = texfetch4(S_tex,iter);
				float4_t Rj   = texfetch4(R_tex,iter);
				float_t  hj   = texfetch1(h_tex,iter);
				float_t  rhoj = texfetch1(rho_tex,iter);
				float_t  prsj = texfetch1(p_tex,iter);
#ifdef Thermal_Conduction_Brookshaw
				float_t Tj = 0.;
				if (thermal_alpha != 0.) {
					Tj = texfetch1(T_tex,iter);
				}
#endif
				float_t is_tool_particle_j = texfetch1(tool_particle_tex,iter);

				float_t volj   = mass/rhoj;
				float_t rhoj21 = 1./(rhoj*rhoj);

				//compute kernel
				float3_t ww = cubic_spline(pi, pj, hi);

				float_t w   = ww.x;
				float_t w_x = ww.y;
				float_t w_y = ww.z;

				// do not run mechanical solver on tool particles (but run the thermal solver)
				if (!(is_tool_particle_i == 1.) && !(is_tool_particle_j == 1.)) {
					//derive vel
					vi_der.x += (vj.x-vi.x)*w_x*volj;
					vi_der.y += (vj.x-vi.x)*w_y*volj;
					vi_der.z += (vj.y-vi.y)*w_x*volj;
					vi_der.w += (vj.y-vi.y)*w_y*volj;

					//derive stress + art stress
					float_t  fab = w/wdeltap;
					fab *= fab;		//to the power of 4
					fab *= fab;

					float3_t R;
					R.x = fab*(Ri.x + Rj.x);
					R.y = fab*(Ri.y + Rj.y);
					R.z = fab*(Ri.z + Rj.z);

					Si_der.x += mass*((Si.x-prsi)*rhoi21 + (Sj.x-prsj)*rhoj21 + R.x)*w_x;
					Si_der.y += mass*(Si.y*rhoi21        + Sj.y*rhoj21        + R.y)*w_y;
					Si_der.z += mass*(Si.y*rhoi21        + Sj.y*rhoj21        + R.y)*w_x;
					Si_der.w += mass*((Si.z-prsi)*rhoi21 + (Sj.z-prsj)*rhoj21 + R.z)*w_y;

					//art visc
					float_t xij = pi.x - pj.x;
					float_t yij = pi.y - pj.y;

					float_t vijx = vi.x - vj.x;
					float_t vijy = vi.y - vj.y;

					float_t vijposij = vijx*xij + vijy*yij;
					float_t rhoij = 0.5*(rhoi+rhoj);

					if (vijposij < 0.) {
						float_t ci   = sqrtf(K/rhoi);
						float_t cj   = sqrtf(K/rhoj);

						float_t cij = 0.5*(ci+cj);
						float_t hij = 0.5*(hi+hj);

						float_t r2ij = xij*xij + yij*yij;
						float_t muij = (hij*vijposij)/(r2ij + eta*eta*hij*hij);
						float_t piij = (-alpha*cij*muij + beta*muij*muij)/rhoij;

						vi_t.x += -mass*piij*w_x;
						vi_t.y += -mass*piij*w_y;
					}

					//xsph
					xi_t.x += -eps*w*mass/rhoij*vijx;
					xi_t.y += -eps*w*mass/rhoij*vijy;

				}

#ifdef Thermal_Conduction_Brookshaw
				//thermal	(brookshaw approximation)
				if (thermal_alpha != 0.) {
					float2_t pj   = texfetch2(pos_tex,iter);
					float_t xij = pi.x - pj.x;
					float_t yij = pi.y - pj.y;
					float_t rij = sqrt(xij*xij + yij*yij);
					if (rij > 1e-8) {
						float_t eijx = xij/rij;
						float_t eijy = yij/rij;
						float_t rij1 = 1./rij;
						T_lapl += 2.0*(mass/rhoj)*(Ti-Tj)*rij1*(eijx*w_x + eijy*w_y);
					}
				}
#endif

			}
		}
	}

	//write back
	S_der[pidx] = Si_der;
	v_der[pidx] = vi_der;

	pos_t[pidx] = xi_t;
	vel_t[pidx] = vi_t;

#ifdef Thermal_Conduction_Brookshaw
	if (thermal_alpha != 0.) {
		T_t[pidx] = thermal_alpha*T_lapl;
	}
#endif
}

void interactions_monaghan(particle_gpu *particles, const int *cells_start, const int *cells_end, int num_cell) {
	//run all interactions in one go
	dim3 dG((particles->N + BLOCK_SIZE-1) / BLOCK_SIZE);
	dim3 dB(BLOCK_SIZE);

	particle_gpu *p = particles;
	int N = p->N;

	cudaBindTexture(0, pos_tex,    p->pos,    sizeof(float_t)*N*2);
	cudaBindTexture(0, vel_tex,    p->vel,    sizeof(float_t)*N*2);
	cudaBindTexture(0, S_tex,      p->S,      sizeof(float_t)*N*4);
	cudaBindTexture(0, R_tex,      p->R,      sizeof(float_t)*N*4);
	cudaBindTexture(0, h_tex,      p->h,      sizeof(float_t)*N);
	cudaBindTexture(0, rho_tex,    p->rho,    sizeof(float_t)*N);
	cudaBindTexture(0, p_tex,      p->p,      sizeof(float_t)*N);
	cudaBindTexture(0, T_tex,      p->T,   sizeof(float_t)*N);
	cudaBindTexture(0, hashes_tex, p->hash,   sizeof(int)*N);
	cudaBindTexture(0, tool_particle_tex, p->tool_particle, sizeof(float_t)*N);

	cudaBindTexture(0, cells_start_tex, cells_start,   sizeof(int)*num_cell);
	cudaBindTexture(0, cells_end_tex,   cells_end,     sizeof(int)*num_cell);

#ifdef Thermal_Conduction_Brookshaw
	do_interactions_monaghan<<<dG,dB>>>(p->blanked, p->v_der, p->S_der, p->pos_t, p->vel_t, p->T_t, p->N);
#else
	do_interactions_monaghan<<<dG,dB>>>(p->blanked, p->v_der, p->S_der, p->pos_t, p->vel_t, p->N);
#endif

	cudaUnbindTexture(pos_tex);
	cudaUnbindTexture(vel_tex);
	cudaUnbindTexture(S_tex);
	cudaUnbindTexture(R_tex);
	cudaUnbindTexture(h_tex);
	cudaUnbindTexture(rho_tex);
	cudaUnbindTexture(T_tex);
	cudaUnbindTexture(p_tex);
	cudaUnbindTexture(hashes_tex);
	cudaUnbindTexture(tool_particle_tex);
	cudaUnbindTexture(cells_start_tex);
	cudaUnbindTexture(cells_end_tex);

	check_cuda_error("interactions monaghan\n");
}

void interactions_heat_pse(particle_gpu *particles, const int *cells_start, const int *cells_end, int num_cell) {
	if (!m_thermal_workpiece) return;

	dim3 dG((particles->N + BLOCK_SIZE-1) / BLOCK_SIZE);
	dim3 dB(BLOCK_SIZE);

	particle_gpu *p = particles;
	int N = p->N;

	cudaBindTexture(0, pos_tex,    p->pos,    sizeof(float_t)*N*2);
	cudaBindTexture(0, h_tex,      p->h,      sizeof(float_t)*N);
	cudaBindTexture(0, rho_tex,    p->rho,    sizeof(float_t)*N);
	cudaBindTexture(0, T_tex,      p->T,      sizeof(float_t)*N);
	cudaBindTexture(0, hashes_tex, p->hash,   sizeof(int)*N);
	cudaBindTexture(0, tool_particle_tex, p->tool_particle,   sizeof(int)*N);

	cudaBindTexture(0, cells_start_tex, cells_start,   sizeof(int)*num_cell);
	cudaBindTexture(0, cells_end_tex,   cells_end,     sizeof(int)*num_cell);

	do_interactions_heat_pse<<<dG,dB>>>(p->blanked, p->T_t, p->N, thermals_workpiece.alpha, thermals_tool.alpha);

	cudaUnbindTexture(pos_tex);
	cudaUnbindTexture(h_tex);
	cudaUnbindTexture(rho_tex);
	cudaUnbindTexture(T_tex);
	cudaUnbindTexture(hashes_tex);
	cudaUnbindTexture(tool_particle_tex);

	cudaUnbindTexture(cells_start_tex);
	cudaUnbindTexture(cells_end_tex);
}

void interactions_setup_geometry_constants(grid_base *g) {
	geom_constants geometry_h;
	geometry_h.nx = g->nx();
	geometry_h.ny = g->ny();
	geometry_h.bbmin_x = g->bbmin_x();
	geometry_h.bbmin_y = g->bbmin_y();
	geometry_h.dx = g->dx();
	cudaMemcpyToSymbol(geometry, &geometry_h, sizeof(geom_constants), 0, cudaMemcpyHostToDevice);
}

void interactions_setup_physical_constants(phys_constants physics_h) {
	cudaMemcpyToSymbol(physics, &physics_h, sizeof(phys_constants), 0, cudaMemcpyHostToDevice);
}

void interactions_setup_corrector_constants(corr_constants correctors_h) {
	cudaMemcpyToSymbol(correctors, &correctors_h, sizeof(corr_constants), 0, cudaMemcpyHostToDevice);
}

void interactions_setup_thermal_constants_workpiece(trml_constants trml_h) {
	thermals_workpiece = trml_h;
	m_thermal_workpiece = trml_h.alpha != 0.;
	if (m_thermal_workpiece) {
		printf("considering thermal diffusion in workpiece\n");
	}

	cudaMemcpyToSymbol(trml, &trml_h, sizeof(trml_constants), 0, cudaMemcpyHostToDevice);
}

void interactions_setup_thermal_constants_tool(trml_constants trml_h, tool *tool) {
	thermals_tool = trml_h;
	m_thermal_tool = trml_h.alpha != 0.;
	if (m_thermal_tool) {
		tool->set_thermal(true);
		printf("considering thermal diffusion from workpiece into tool\n");
		cudaMemcpyToSymbol(trml_tool, &trml_h, sizeof(trml_constants), 0, cudaMemcpyHostToDevice);
	}
}
