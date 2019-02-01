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

#include "tool_wear.h"

void tool_wear::eval_usui(const particle_gpu *particles, float_t dt, float_t &usui_min, float_t &usui_max, float_t &usui_avg) {
	const unsigned int N_part = particles->N;

	cudaMemcpy(m_vel_buf,    particles->vel, sizeof(float_t)*2*N_part, cudaMemcpyDeviceToHost);
	cudaMemcpy(m_fc_buf,     particles->fc,  sizeof(float_t)*2*N_part, cudaMemcpyDeviceToHost);
	cudaMemcpy(m_ft_buf,     particles->ft,  sizeof(float_t)*2*N_part, cudaMemcpyDeviceToHost);
	cudaMemcpy(m_h_buf,      particles->h,   sizeof(float_t)*N_part,   cudaMemcpyDeviceToHost);
	cudaMemcpy(m_normal_buf, particles->n,   sizeof(float_t)*2*N_part, cudaMemcpyDeviceToHost);
	cudaMemcpy(m_T_buf,      particles->T,   sizeof(float_t)*N_part,   cudaMemcpyDeviceToHost);

	usui_min   = FLT_MAX;
	usui_max   = 0.;
	float_t usui_total = 0.;
	unsigned int num_in_contact = 0;

	for (unsigned int i = 0; i < N_part; i++) {
		bool in_contact = fabs(m_fc_buf[i].x) > 0.  || fabs(m_fc_buf[i].y) > 0. || fabs(m_ft_buf[i].x) > 0. || fabs(m_fc_buf[i].y) > 0.;

		if (in_contact) {
			num_in_contact++;

			//compute relative velocity
			vec2_t vs(m_vel_buf[i].x, m_vel_buf[i].y);
			vec2_t n(m_normal_buf[i].x, m_normal_buf[i].y);
			vec2_t v = vs-m_tool_vel;
			vec2_t vr = v - v*n;

			float_t vmag = glm::length(vr);
			float_t fmag = glm::length(vec2_t(m_fc_buf[i].x, m_fc_buf[i].y));

			//calcualte pressure. assuming particle spans a disk of radius 2*h on tool
			float_t A = 2*m_h_buf[i]*M_PI;
			float_t p = fmag/A;

			float_t wear_usui_dot = m_K*exp(m_alpha/m_T_buf[i])*p*vmag;
			usui_total += wear_usui_dot;
			usui_min = fmin(usui_min, wear_usui_dot);
			usui_max = fmax(usui_max, wear_usui_dot);
		}
	}

	usui_min = (num_in_contact == 0) ? 0. : usui_min;
	usui_avg = (num_in_contact == 0) ? 0. : usui_total/num_in_contact;
	m_accum_wear += dt*usui_avg;
}

float_t tool_wear::get_accum_wear() const {
	return m_accum_wear;
}

tool_wear::tool_wear(float_t usui_K, float_t usui_alpha, unsigned int N_part, phys_constants physical_constants, vec2_t tool_vel)
: m_K(usui_K), m_alpha(usui_alpha), m_physical_constants(physical_constants), m_tool_vel(tool_vel) {
	m_vel_buf    = new float2_t[N_part];
	m_fc_buf     = new float2_t[N_part];
	m_ft_buf     = new float2_t[N_part];
	m_h_buf      = new float_t[N_part];
	m_T_buf      = new float_t[N_part];
	m_normal_buf = new float2_t[N_part];
}
