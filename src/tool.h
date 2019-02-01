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

//representation of the tool on the cpu
//used for things like
//	construction of the tool given rake angle, clearance angle and fillet
//	adjusting the tool to a target feed
//	sampling the tool with particles for the thermal solver
//	output to vtk / txt files

#ifndef TOOL_H_
#define TOOL_H_

#include <glm/glm.hpp>
#include <vector>
#include <stdio.h>
#include <assert.h>

#include "particle_gpu.h"
#include "geometric_primitives.h"

class tool {

public:

	bbox safe_bb(float_t safety = 0.011) const;

	// return lowest point of tool
	// 		convenient to measure depth of cut (aka feed)
	float_t low() const;

	// return friction coefficient
	float_t mu() const;

	// return front of tool
	float_t front() const;

	// move tool with it's velocity
	void update_tool(float_t dt);

	// set velocity of tool
	void set_vel(vec2_t vel);

	// tool thermals
	void set_thermal(bool thermal);
	bool thermal() const;

	// get velocity of tool
	vec2_t get_vel() const;

	// returns center (in the sense of center of gravity) of the tool
	//		used for debugging purposes
	vec2_t center() const;

	//chamfer debugging
	void get_chamfer_data(vec2_t &p, float_t &r) const;
	void set_chamfer(vec2_t cp, float_t r, float_t t1, float_t t2);
	void set_chamfer_debug(bool chamfer_debug);

	//needed to pass stuff to gpu
	circle_segment *get_fillet() const;
	std::vector<segment> get_segments() const;

	// print to file
	void print(FILE *fp);
	void print(unsigned int step, const char *folder = "results");
	void print();

	// sample the tool with tool particles
	//		returns sample positions on CPU
	std::vector<float2_t> sample_tool(float_t dx, float_t T_init, float_t rho_init, float_t h);

	// construct tool given by four points and a fillet radius
	tool(vec2_t tl, vec2_t tr, vec2_t br, vec2_t bl, float_t r, float_t mu_fric);

	// construct tool given by four points (tool is perfectly sharp)
	tool(vec2_t tl, vec2_t tr, vec2_t br, vec2_t bl, float_t mu_fric);

	// construct tool given by a reference point, length and height
	// as well as rake and clearance angle (measured from vertically
	// downwards and horizontally leftwards, respectively), and
	// fillet radius r
	// angles are in degrees
	tool(vec2_t tl, float_t length, float_t height,
			float_t rake_angle, float_t clearance_angle,
			float_t r, float_t mu_fric);

	// construct tool given by a reference point, length and height
	// as well as rake and clearance angle (measured from vertically
	// downwards and horizontally leftwards, respectively), the tool
	// is percectly sharp
	// angles are in degrees
	tool(vec2_t tl, float_t length, float_t height,
			float_t rake_angle, float_t clearance_angle, float_t mu_fric);

	tool();

private:
	// fit a fillet on line lm to l1 with radius r
	// alternatively: find a point on lm with perpendicular distance r to l1
	vec2_t fit_fillet(float_t r, line lm, line l1) const;

	// construct segments from list of points
	void construct_segments(std::vector<vec2_t> list_p);

	// intersect line tr to br and line br to bl with fillet of radius r
	// (4 points -> 5 points) and set fillet
	std::vector<vec2_t> construct_points_and_fillet(vec2_t tl, vec2_t tr, vec2_t br, vec2_t bl, float_t r);

	// velocity of tool
	vec2_t m_velocity = vec2_t(0.,0.);

	// friction coefficient
	float_t m_mu = 0.;

	// front of the tool
	float_t m_front = 0.;

	// thermals considered?
	bool m_thermal = false;

	// componentsm
	circle_segment *m_fillet = 0;
	std::vector<segment> m_segments;

	// if true tool consists of circle segment only
	bool m_chamfer_debug = false;

	bool inside(float2_t pos) const;
};

#endif /* TOOL_H_ */
