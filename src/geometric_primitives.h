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

//this module contains geometrical primitives on the CPU (lines, segments etc.)
//this is used to sample the tool with particles on the CPU

#ifndef GEOMETRIC_PRIMITIVES_H_
#define GEOMETRIC_PRIMITIVES_H_

#include "types.h"

struct line {
	float_t a = 0.;
	float_t b = 0.;
	bool vertical = false;

	//return points closest to xq on this line
	vec2_t closest_point(vec2_t xq) const;

	//return intersection point
	vec2_t intersect(line l) const;

	line(float_t a, float_t b, bool vertical);
	line();
	// construct line from two points
	line(vec2_t p1, vec2_t p2);
};

struct segment {
	vec2_t left;		// left end
	vec2_t right;		// right end
	line l;			// line representation
	vec2_t n;			// normal

	segment(vec2_t left, vec2_t right);
	segment();
	float_t length() const;
};

struct circle_segment {
	float_t r  = 0.;				//radius
	float_t t1 = 0.;				//starting angle
	float_t t2 = 0.;				//end angle
	vec2_t p;			    //center

	circle_segment(float_t r, float_t t1, float_t t2, vec2_t p);
	circle_segment(vec2_t p1, vec2_t p2, vec2_t p3);
	circle_segment();
};

struct bbox {
	float_t bbmin_x = 0.;
	float_t bbmax_x = 0.;
	float_t bbmin_y = 0.;
	float_t bbmax_y = 0.;

	bool in(vec2_t qp);

	// checks if bboy axtually spans a reasonable area
	bool valid() const;

	bbox();
	bbox(vec2_t p1, vec2_t p2);
	bbox(float_t bbmin_x, float_t bbmax_x, float_t bbmin_y, float_t bbmax_y);
};



#endif /* GEOMETRIC_PRIMITIVES_H_ */
