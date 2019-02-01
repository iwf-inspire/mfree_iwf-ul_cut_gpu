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

//module containing the common interface for spatial hashing

#ifndef GRID_H_
#define GRID_H_

#include "particle_gpu.h"
#include "tool.h"
#include "types.h"

#include <thrust/extrema.h>
#include <thrust/device_ptr.h>

#include <stdio.h>

class grid_base {

protected:
	//geometry
	double m_dx = 0.;
	double m_lx = 0., m_ly = 0.;
	unsigned int m_nx = 0, m_ny = 0;
	unsigned int m_num_cell = 0;

	double m_bbmin_x = 0., m_bbmax_x = 0.;
	double m_bbmin_y = 0., m_bbmax_y = 0.;

	int m_max_cell = 0;
	int m_num_part = 0;

	bool m_geometry_locked = false;

	void set_geometry(float2_t bbmin, float2_t bbmax, float_t h);

public:
	int nx() const;
	int ny() const;
	float_t bbmin_x() const;
	float_t bbmin_y() const;
	float_t bbmax_x() const;
	float_t bbmax_y() const;
	float_t dx() const;
	bool is_locked() const;
	int num_cell() const;

	void assign_hashes(particle_gpu *partilces, tool *tool) const;
	void update_geometry(particle_gpu *particles, tool *tool, float_t kernel_width = 2.0);

	virtual void sort(particle_gpu *particles, tool *tool) const = 0;
	virtual void get_cells(particle_gpu *particles, int *cell_start, int *cell_end) = 0;

	grid_base(int max_cell, int num_part);

	//locks geometry! calls to update geometry will NOT work
	grid_base(int num_part, float2_t bbmin, float2_t bbmax, float_t h);
	virtual ~grid_base();
};

#endif /* GRID_H_ */
