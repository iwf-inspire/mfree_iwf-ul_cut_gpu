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

#include "debug.h"

void dump_state(particle_gpu *particles, tool *tool, int step) {
	int n = particles->N;

	int      *h_idx = new int[n];
	float2_t *h_pos = new float2_t[n];
	float2_t *h_vel = new float2_t[n];
	float_t  *h_rho = new float_t[n];
	float_t  *h_h   = new float_t[n];
	float_t  *h_p   = new float_t[n];
	float4_t *h_S   = new float4_t[n];
	float_t  *h_T   = new float_t[n];
	float_t  *h_eps = new float_t[n];

	cudaMemcpy(h_idx, particles->idx,    sizeof(int)*n,      cudaMemcpyDeviceToHost);
	cudaMemcpy(h_pos, particles->pos,    sizeof(float2_t)*n, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_vel, particles->vel,    sizeof(float2_t)*n, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_rho, particles->rho,    sizeof(float_t)*n,  cudaMemcpyDeviceToHost);
	cudaMemcpy(h_h,   particles->h,      sizeof(float_t)*n,  cudaMemcpyDeviceToHost);
	cudaMemcpy(h_p,   particles->p,      sizeof(float_t)*n,  cudaMemcpyDeviceToHost);
	cudaMemcpy(h_S,   particles->S,      sizeof(float4_t)*n, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_T,   particles->T,      sizeof(float_t)*n,  cudaMemcpyDeviceToHost);
	cudaMemcpy(h_eps, particles->eps_pl, sizeof(float_t)*n,  cudaMemcpyDeviceToHost);

	char buf[256];
	sprintf(buf, "results/dump_%06d.txt", step);
	FILE *fp = fopen(buf, "w+");


	for (int i = 0; i < n; i++) {
		fprintf(fp, "%.9g %.9g %.9g %.9g %.9g %.9g %.9g %.9g %.9g %.9g %.9g\n",
				h_pos[i].x, h_pos[i].y, h_vel[i].x, h_vel[i].y, h_rho[i], h_p[i], h_S[i].x, h_S[i].y, h_S[i].z, h_T[i], h_eps[i]);
	}

	delete[] h_idx;
	delete[] h_pos;
	delete[] h_vel;
	delete[] h_rho;
	delete[] h_h;
	delete[] h_p;
	delete[] h_S;
	delete[] h_T;
	delete[] h_eps;

	fclose(fp);
}
