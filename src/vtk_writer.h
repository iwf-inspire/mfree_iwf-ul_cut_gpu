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

//output to vtk legacy format
//	https://www.google.com/search?q=paraview

#ifndef VTK_WRITER_H_
#define VTK_WRITER_H_

#include "particle_gpu.h"
#include "types.h"
#include "tool.h"

void vtk_writer_write(const particle_gpu *particles, int step);
void vtk_writer_write(const tool* tool, int step);

#endif /* VTK_WRITER_H_ */
