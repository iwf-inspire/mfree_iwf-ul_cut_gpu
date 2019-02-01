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

//kernel functions (cubic spline and PSE kernel)

#ifndef KERNELS_CUH_
#define KERNELS_CUH_

#include <cuda.h>
#include <cuda_runtime.h>

#include "types.h"

#define PI_FAC 0.454728408833987

__host__ __device__  float3_t cubic_spline(float2_t posi, float2_t posj, float_t hi) {

	float3_t w;
	w.x = 0.;
	w.y = 0.;
	w.z = 0.;

	float_t xij = posi.x-posj.x;
	float_t yij = posi.y-posj.y;

	float_t rr2=xij*xij+yij*yij;
	float_t h1 = 1/hi;
	float_t fourh2 = 4*hi*hi;
	if(rr2>=fourh2 || rr2 < 1e-8) {
		return w;
	}

	float_t der;
	float_t val;

	float_t rad=sqrtf(rr2);
	float_t q=rad*h1;
	float_t fac = PI_FAC*h1*h1;

	const bool radgt=q>1;
	if (radgt) {
		float_t _2mq  = 2-q;
		float_t _2mq2 = _2mq*_2mq;
		val = 0.25*_2mq2*_2mq2*_2mq2;
		der = -0.75*_2mq2 * h1/rad;
	} else {
		val = 1 - 1.5*q*q*(1-0.5*q);
		der = -3.0*q*(1-0.75*q) * h1/rad;
	}

	w.x = val*fac;
	w.y = der*xij*fac;
	w.z = der*yij*fac;

	return w;
}

__host__ __device__  float_t lapl_pse(float2_t posi, float2_t posj, float_t hi) {
	float_t xi = posi.x;
	float_t yi = posi.y;

	float_t xj = posj.x;
	float_t yj = posj.y;

	float_t xij = xi-xj;
	float_t yij = yi-yj;

	float_t xx = sqrt(xij*xij + yij*yij);

	float_t h2 = hi*hi;
	float_t h4 = h2*h2;
	float_t w2_pse  = +4./(h4*M_PI)*exp(-xx*xx/(h2));

	return w2_pse;
}

#endif /* KERNELS_CUH_ */
