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

#include "grid.h"

struct compare_float2_x {
  __host__ __device__
  bool operator()(float2_t lhs, float2_t rhs) {
    return lhs.x < rhs.x;
  }
};

struct compare_float2_y {
  __host__ __device__
  bool operator()(float2_t lhs, float2_t rhs) {
    return lhs.y < rhs.y;
  }
};

__global__ static void compute_hashes(const float_t *__restrict__ blanked, const float2_t *__restrict__ pos, int * __restrict__ hashes, int count,
								float_t dx, int ny, float_t bbmin_x, float_t bbmin_y) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < count) {
		if (blanked[idx] == 1.) {
			return;
		}

		float_t _px = pos[idx].x;
		float_t _py = pos[idx].y;

		int ix = (_px - bbmin_x)/dx;
		int iy = (_py - bbmin_y)/dx;

		hashes[idx] = ix*ny + iy;
	}
}

int grid_base::nx() const {
	return m_nx;
}

int grid_base::ny() const{
	return m_ny;
}

float_t grid_base::bbmin_x() const {
	return m_bbmin_x;
}

float_t grid_base::bbmin_y() const {
	return m_bbmin_y;
}

float_t grid_base::bbmax_x() const {
	return m_bbmax_x;
}

float_t grid_base::bbmax_y() const {
	return m_bbmax_y;
}

float_t grid_base::dx() const {
	return m_dx;
}

bool grid_base::is_locked() const {
	return m_geometry_locked;
}

int grid_base::num_cell() const {
	return m_num_cell;
}

void grid_base::assign_hashes(particle_gpu *particles, tool *tool) const {
	dim3 dG((particles->N + BLOCK_SIZE-1) / BLOCK_SIZE);
	dim3 dB(BLOCK_SIZE);

	compute_hashes<<<dG,dB>>>(particles->blanked, particles->pos, particles->hash, particles->N,
			m_dx, m_ny, m_bbmin_x, m_bbmin_y);
}

void grid_base::update_geometry(particle_gpu *particles, tool *tool, float_t kernel_width) {
	if (m_geometry_locked) {
		return;
	}

	unsigned int N = particles->N;

	thrust::device_ptr<float2_t> t_pos(particles->pos);
	thrust::device_ptr<float_t> t_h(particles->h);

	thrust::device_ptr<float2_t> minx = thrust::min_element(t_pos, t_pos+N, compare_float2_x());
	thrust::device_ptr<float2_t> miny = thrust::min_element(t_pos, t_pos+N, compare_float2_y());
	thrust::device_ptr<float2_t> maxx = thrust::max_element(t_pos, t_pos+N, compare_float2_x());
	thrust::device_ptr<float2_t> maxy = thrust::max_element(t_pos, t_pos+N, compare_float2_y());

	thrust::device_ptr<float_t> maxh = thrust::max_element(t_h, t_h+N);

	//copy 4 floats individually back to host here
	//		could be optimized since values are not really needed on host and are sent back to gpu via argument passing
	float2_t f2_bbmin_x = minx[0];
	float2_t f2_bbmin_y = miny[0];
	float2_t f2_bbmax_x = maxx[0];
	float2_t f2_bbmax_y = maxy[0];

	m_bbmin_x = f2_bbmin_x.x - 1e-6;
	m_bbmin_y = f2_bbmin_y.y - 1e-6;
	m_bbmax_x = f2_bbmax_x.x + 1e-6;
	m_bbmax_y = f2_bbmax_y.y + 1e-6;

	m_dx = maxh[0]*kernel_width/2;

	m_lx = m_bbmax_x - m_bbmin_x;
	m_ly = m_bbmax_y - m_bbmin_y;

	m_nx = ceil(m_lx/m_dx);
	m_ny = ceil(m_ly/m_dx);
	m_num_cell = m_nx*m_ny;

	assert(m_num_cell < m_max_cell);
}


void grid_base::set_geometry(float2_t bbmin, float2_t bbmax, float_t h) {
	m_bbmin_x = bbmin.x - 1e-6;
	m_bbmin_y = bbmin.y - 1e-6;
	m_bbmax_x = bbmax.x + 1e-6;
	m_bbmax_y = bbmax.y + 1e-6;

	float_t kernel_width = 2.;
	m_dx = h*kernel_width/2.;

	m_lx = m_bbmax_x - m_bbmin_x;
	m_ly = m_bbmax_y - m_bbmin_y;

	m_nx = ceil(m_lx/m_dx);
	m_ny = ceil(m_ly/m_dx);

	m_num_cell = m_nx*m_ny;
	m_max_cell = m_num_cell;

	m_geometry_locked = true;
}

grid_base::grid_base(int max_cell, int num_part) :
		m_max_cell(max_cell), m_num_cell(max_cell), m_num_part(num_part)
{}

grid_base::grid_base(int num_part, float2_t bbmin, float2_t bbmax, float_t h) {
	m_num_part = num_part;
	set_geometry(bbmin, bbmax, h);
}

grid_base::~grid_base() {
}
