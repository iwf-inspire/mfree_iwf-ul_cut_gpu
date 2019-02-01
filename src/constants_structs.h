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

//this module defines collections of constants (e.g. physical constants, correction constants etc.)

#ifndef CONSTANTS_STRUCTS_H_
#define CONSTANTS_STRUCTS_H_

#include "types.h"

#include <cstring>

struct phys_constants {
	float_t E;
	float_t nu;
	float_t rho0;
	float_t K;
	float_t G;
	float_t mass;
	float_t mass_tool;
};

phys_constants make_phys_constants();

struct trml_constants {
	float_t cp; //thermal capacity
	float_t tq;	//taylor guinnie
	float_t k;	//thermal conductivity
	float_t alpha;	//thermal diffusitivty
	float_t T_init;	//initial temperature
	float_t eta;	//fraction of frictional power turned to heat
};

trml_constants make_trml_constants();

struct corr_constants {
	float_t wdeltap;
	float_t stresseps;
	float_t xspheps;
	float_t alpha;
	float_t beta;
	float_t eta;
};

corr_constants make_corr_constants();

struct joco_constants {
	float_t A;
	float_t B;
	float_t C;
	float_t n;
	float_t m;
	float_t Tmelt;
	float_t Tref;
	float_t eps_dot_ref;

	float_t tanh_a;
	float_t tanh_b;
	float_t tanh_c;
	float_t tanh_d;
};

joco_constants make_joco_constants();

struct geom_constants {
	int nx;
	int ny;
	float_t bbmin_x;
	float_t bbmin_y;
	float_t dx;
};

geom_constants make_geom_constants();

#endif /* CONSTANTS_STRUCTS_H_ */
