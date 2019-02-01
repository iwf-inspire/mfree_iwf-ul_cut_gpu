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

#include "output.h"

// float2 + struct
struct add_float2 {
    __device__ float2_t operator()(const float2_t& a, const float2_t& b) const {
        float2_t r;
        r.x = a.x + b.x;
        r.y = a.y + b.y;
        return r;
    }
 };


float2_t report_force(particle_gpu *particles) {
	thrust::device_ptr<float2_t> t_fc(particles->fc);
	thrust::device_ptr<float2_t> t_ft(particles->ft);
	float2_t ini;
	ini.x = 0.;
	ini.y = 0.;
	ini = thrust::reduce(t_fc, t_fc + particles->N, ini, add_float2());
	return thrust::reduce(t_ft, t_ft + particles->N, ini, add_float2());
}
