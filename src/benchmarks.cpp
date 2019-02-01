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

#include "benchmarks.h"

float_t global_dt = 0.;
float_t global_t_final = 0.;

particle_gpu *setup_impact(int nbox, grid_base **grid) {
	phys_constants phys = make_phys_constants();
	corr_constants corr = make_corr_constants();
	joco_constants joco = make_joco_constants();

	float_t ro  = 0.2;
	float_t dx  = 2*ro/(nbox-1);
	float_t hdx = 1.7;

	//material is steel 4430
	phys.E    = 200e9;
	phys.nu   = 0.29;
	phys.rho0 = 7830.0;
	phys.G    = phys.E/(2.*(1.+phys.nu));
	phys.K    = 2.0*phys.G*(1+phys.nu)/(3*(1-2*phys.nu));
	phys.mass = dx*dx*phys.rho0;

	joco.A		= 792.0e6;
	joco.B		= 510.0e6;
	joco.C		= 0.014;
	joco.m		= 1.03;
	joco.n		= 0.26;
	joco.Tref	= 273.0;
	joco.Tmelt	= 1793.0;
	joco.eps_dot_ref = 1;

	//artificial viscosity, XSPH and artificial stress constants
	corr.alpha   = 1.;
	corr.beta    = 1.;
	corr.eta     = 0.1;
	corr.xspheps = 0.5;
	corr.stresseps = 0.3;
	{
		float_t h1   = 1./(hdx*dx);
		float_t q    = dx*h1;
		float_t fac  = 10*(M_1_PI)/7.0*h1*h1;
		corr.wdeltap = fac*(1 - 1.5*q*q*(1-0.5*q));
	}

	//generate particles
	float_t spacing = ro + 0.05;

	int guess_number = 2*(nbox*nbox + 20*nbox*nbox);
	float2_t *pos = new float2_t[guess_number];

	//flyer
	int part_iter = 0;
	for (int i = 0; i < nbox; i++) {
		for (int j = 0; j < nbox; j++) {
			float_t px = -ro+i*dx; float_t py = -ro+j*dx;
			float_t dist = sqrt(px*px + py*py);
			if (dist < ro) {
				pos[part_iter].x = px-spacing;
				pos[part_iter].y = py;
				part_iter++;
			}
		}
	}

	int n_flyer = part_iter;

	//wall
	for (int i = 0; i < nbox/2; i++) {
		for (int j = 0; j < 10*nbox; j++) {
			if (j*dx-1. > 1.) {
				continue;
			}
			pos[part_iter].x = i*dx;
			pos[part_iter].y = j*dx-1.;
			part_iter++;
		}
	}

	int n = part_iter;

	float_t vel_flyer = 1e3;
	global_dt = 0.3*hdx*dx/(sqrt(phys.K/phys.rho0) + sqrt(vel_flyer));
	global_t_final = 2*ro/vel_flyer;

	printf("time step used %e\n",   global_dt);
	printf("calculating with %d particles\n", part_iter);

	//initial and boundary conditions
	float2_t *vel = new float2_t[n];
	float_t  *rho = new float_t[n];
	float_t  *h   = new float_t[n];
	float_t  *fixed   = new float_t[n];
	float_t  *T   = new float_t[n];

	for (int i = 0; i < n; i++) {
		rho[i] = phys.rho0;
		h[i] = hdx*dx;
		vel[i].x = 0.;
		vel[i].y = 0.;
		if (i < n_flyer) {
			vel[i].x = 1e3;
		}

		//fix top and bottom of wall
		if (pos[i].y < -1+dx/2 || pos[i].y > 1-dx/2) {
			fixed[i] = 1.;
		} else {
			fixed[i] = 0.;
		}

		T[i] = joco.Tref;
	}

	//set up grid for spatial hashing
	//	- grid rothlin saves memory but is slower than grid green
	//	- each grid can be configured to adapt to the solution domain or to stay fixed
	//	- fixed grids work considerably faster than adapting ones

	*grid = new grid_gpu_green(10*n, n);
//	*grid = new grid_gpu_green(n, make_float2_t(-.5,-1.2), make_float2_t(1,1.2), hdx*dx);
//	*grid = new grid_gpu_rothlin(10*n, n);
//	*grid = new grid_gpu_rothlin(n, make_float2_t(-.5,-1.2), make_float2_t(1,1.2), hdx*dx);

	//communicate constants to interactions and actions
	actions_setup_corrector_constants(corr);
	actions_setup_physical_constants(phys);
	actions_setup_johnson_cook_constants(joco);

	interactions_setup_corrector_constants(corr);
	interactions_setup_physical_constants(phys);
	interactions_setup_geometry_constants(*grid);

	assert(check_cuda_error());

	return new particle_gpu(pos, vel, rho, T, h, fixed, n);
}

particle_gpu *setup_rings(int nbox, grid_base **grid) {
	phys_constants phys = make_phys_constants();
	corr_constants corr = make_corr_constants();

	//problem dimensions (monaghan & gray)
	float_t ri = 0.03;
	float_t ro = 0.04;
	float_t spacing = ro + 0.001;

	float_t dx = 2*ro/(nbox-1);
	float_t hdx = 1.7;

	//virtual material that behaves rubber like
	phys.E    = 1e7;
	phys.nu   = 0.4;
	phys.rho0 = 1.;
	phys.G    = phys.E/(2.*(1.+phys.nu));
	phys.K    = 2.0*phys.G*(1+phys.nu)/(3*(1-2*phys.nu));
	phys.mass = dx*dx*phys.rho0;

	//artificial viscosity, XSPH and artificial stress constants
	corr.alpha   = 1.;
	corr.beta    = 1.;
	corr.eta     = 0.1;
	corr.xspheps = 0.5;
	corr.stresseps = 0.3;
	{
		float_t h1   = 1./(hdx*dx);
		float_t q    = dx*h1;
		float_t fac  = 10*(M_1_PI)/7.0*h1*h1;
		corr.wdeltap = fac*(1 - 1.5*q*q*(1-0.5*q));
	}

	//generate samples in rings
	float2_t *pos = new float2_t[nbox*nbox];
	int part_iter = 0;
	for (int i = 0; i < nbox; i++) {
		for (int j = 0; j < nbox; j++) {
			float_t px = -ro+i*dx; float_t py = -ro+j*dx;
			float_t dist = sqrt(px*px + py*py);
			if (dist < ro && dist >= ri) {
				pos[part_iter].x = px-spacing;
				pos[part_iter].y = py;
				part_iter++;
				pos[part_iter].x = px+spacing;
				pos[part_iter].y = py;
				part_iter++;
			}
		}
	}

	//ring velocity (relative impact velocity = 2*ring velocity)
	float_t vel_ring = 170.;
	global_t_final = 2.5*ro/vel_ring;

	//CFL based choice of time step
	global_dt = 0.5*hdx*dx/(sqrt(phys.K/phys.rho0) + sqrt(vel_ring));
	printf("timestep chosen: %e\n", global_dt);

	assert(part_iter < nbox*nbox);
	int n = part_iter;

	//set up grid for spatial hashing
	//	- grid rothlin saves memory but is slower than grid green
	//	- each grid can be configured to adapt to the solution domain or to stay fixed
	//	- fixed grids work considerably faster than adapting ones

//	*grid = new grid_gpu_green(10*n, n);
	*grid = new grid_gpu_green(n, make_float2_t(-5*ro, -2*ro), make_float2_t(5*ro, 2*ro), hdx*dx);
//	*grid = new grid_gpu_rothlin(10*n, n);
//	*grid = new grid_gpu_rothlin(n, make_float2_t(-5*ro, -2*ro), make_float2_t(5*ro, 2*ro), hdx*dx);

	printf("calculating with %d particles\n", part_iter);

	//set remainder of initial conditions
	float2_t *vel = new float2_t[n];
	float_t  *h   = new float_t[n];
	float_t  *rho = new float_t[n];

	for (int i = 0; i < n; i++) {
		rho[i] = phys.rho0;
		h[i] = hdx*dx;
		vel[i].x = (pos[i].x < 0) ? 200 : -200;
		vel[i].y = 0.;
	}

	//commmunicate constants to actions, interactions
	actions_setup_corrector_constants(corr);
	actions_setup_physical_constants(phys);

	interactions_setup_corrector_constants(corr);
	interactions_setup_physical_constants(phys);
	interactions_setup_geometry_constants(*grid);

	particle_gpu *particles = new particle_gpu(pos, vel, rho, h, n);

	assert(check_cuda_error());
	return particles;
}

particle_gpu *setup_ref_cut(int ny, grid_base **grid, float_t rake, float_t clear, float_t chamfer, float_t speed, float_t feed) {
	phys_constants phys = make_phys_constants();
	corr_constants corr = make_corr_constants();
	joco_constants joco = make_joco_constants();
	trml_constants trml_wp = make_trml_constants();
	trml_constants trml_tool = make_trml_constants();

	bool use_thermals = true;		//set thermal constants?
	bool sample_tool  = true;		//expand thermal solver to tool?

	//dimensions of work piece
	float_t hi_x = 0.200; float_t hi_y =  0.060;
	float_t lo_x = 0.005; float_t lo_y =  hi_y-3*feed;

	float_t cutting_distance = 0.1;	//cut 1 mm of material

	float_t dy = (hi_y-lo_y)/(ny-1);
	float_t dx = dy;
	int nx = (hi_x-lo_x)/dx;

	//h = hdx*dx
	float_t hdx = 1.5;

	//Ti6Al4v according to Johnson 85
	phys.E    = 1.1;
	phys.nu   = 0.35;
	phys.rho0 = 4.429998;
	phys.G    = phys.E/(2.*(1.+phys.nu));
	phys.K    = 2.0*phys.G*(1+phys.nu)/(3*(1-2*phys.nu));
	phys.mass = dx*dx*phys.rho0;

	joco.A = 0.0086200;
	joco.B = 0.0033100;
	joco.C = 0.0100000;
	joco.m = 0.8;
	joco.n = 0.34;
	joco.Tref = 300.;
	joco.Tmelt	= 1836.0000;
	joco.eps_dot_ref = 1e-6;

	//constants for TANH JC model for Ti6Al4V according to Ducobu 2017
	//		they are only used if according model is set in plasticity.h
	joco.tanh_a = 1.1;
	joco.tanh_b = 0.4;
	joco.tanh_c = 12.;
	joco.tanh_d = 1.;

	float_t rho_tool = 15.25;
	phys.mass_tool = dx*dx*rho_tool;
	if (use_thermals) {
		// https://www.azom.com/properties.aspx?ArticleID=1203
		trml_wp.cp = 553*1e-8;			// Heat Capacity
		trml_wp.tq = 0.9;				// Taylor-Quinney Coefficient
		trml_wp.k  = 7.1*1e-13;			// Thermal Conduction
		trml_wp.alpha = trml_wp.k/(phys.rho0*trml_wp.cp);	// Thermal diffusivity
		trml_wp.eta = 0.9;
		trml_wp.T_init = joco.Tref;

		//www.azom.com/properties.aspx?ArticleID=1203
		trml_tool.cp = 292*1e-08;
		trml_tool.tq = .0;
		trml_tool.k  = 88*1e-13;
		trml_tool.alpha = trml_tool.k/(rho_tool*trml_tool.cp);
		trml_tool.T_init = joco.Tref;
	}

	//artificial viscosity, XSPH and artificial stress constants
	corr.alpha     = 1.;
	corr.beta      = 1.;
	corr.eta       = 0.1;
	corr.xspheps   = 0.5;
	corr.stresseps = 0.3;
	{
		float_t h1   = 1./(hdx*dx);
		float_t q    = dx*h1;
		float_t fac  = 10*(M_1_PI)/7.0*h1*h1;
		corr.wdeltap = fac*(1 - 1.5*q*q*(1-0.5*q));
	}

	//generate particles in work piece
	int n = nx*ny;
	float2_t *pos = new float2_t[n];
	int part_iter = 0;

    float_t min_x =  FLT_MAX;
	float_t max_x = -FLT_MAX;

    float_t min_y =  FLT_MAX;
	float_t max_y = -FLT_MAX;

	for (int i = 0; i < nx; i++) {
		for (int j = 0; j < ny; j++) {
			pos[part_iter].x = i*dx+lo_x;
			pos[part_iter].y = j*dx+lo_y;

			max_x = fmax(max_x, pos[part_iter].x);
			min_x = fmin(min_x, pos[part_iter].x);
			max_y = fmax(max_y, pos[part_iter].y);
			min_y = fmin(min_y, pos[part_iter].y);

            part_iter++;
		}
	}

	//remaining initial and boundary conditions
	float2_t *vel   = new float2_t[n];
	float_t  *rho   = new float_t[n];
	float_t  *h     = new float_t[n];
	float_t  *T     = new float_t[n];
	float_t  *fixed = new float_t[n];
	float_t  *tool_p = new float_t[n];

	for (int i = 0; i < n; i++) {
		rho[i] = phys.rho0;
		h[i] = hdx*dx;
		vel[i].x = 0.;
		vel[i].y = 0.;
		T[i] = joco.Tref;
		//fix bottom
		fixed[i] = (pos[i].y < min_y+0.5*dx) ? 1. : 0.;
		//fix left hand side below blade
		fixed[i] = fixed[i] || (pos[i].x < min_x + dx/2 && pos[i].y < min_y + 0.5*(hi_y-lo_y));
		//fix all of right hand side
		fixed[i] = fixed[i] || (pos[i].x > max_x - dx);

		tool_p[i] = 0.;
	}

	//tool velocity and friction constant
	float_t vel_tool = (speed == 0.f) ? 83.333328*1e-5 : speed; //500m/min
	float_t mu_fric = 0.35;

	//-----------------------------------------

	//tool dimensions copied from ruttimanns FEM simualations
	float_t nudge = 0.004;
	glm::dvec2 tl(-0.05      + nudge, 0.0986074);

	float_t l_tool = -0.0086824 - -0.05;
	float_t h_tool =  0.0986074 - 0.0555074;

	tool *t = new tool(tl, l_tool, h_tool, rake, clear, chamfer, mu_fric);

	//move tool to target feed
	{
		if (t->front() > lo_x) {
			float_t distance = t->front() - lo_x;
			float_t time = distance/vel_tool;
			t->set_vel(glm::dvec2(vel_tool, 0.));
			t->update_tool(-time);
		}

		float_t target_feed = feed;
		float_t current_feed = hi_y - t->low();
		float_t dist_to_target_feed = fabs(current_feed - target_feed);
		float_t correction_time = dist_to_target_feed / vel_tool;
		float_t sign = (current_feed > target_feed) ? 1 : -1.;
		t->set_vel(glm::dvec2(0., vel_tool));
		t->update_tool(correction_time*sign);
	}

	//set final tool constants
	global_tool = t;
	t->set_vel(glm::dvec2(vel_tool, 0.));
	float_t contact_alpha = 0.1;					// NOTE, IMPORTANT: needs to be
													// reduced if very small timesteps are necessary
	tool_gpu_set_up_tool(t, contact_alpha, phys);

	//sample tool with particles if desired. these particles are only subject to the thermal solver and are ignored by the mechanical solver
	int n_tool = 0;
	if (sample_tool) {
		std::vector<float2_t> samples = t->sample_tool(dx, joco.Tref, phys.rho0, hdx);
		float_t bbmin_tool_x =  FLT_MAX;
		float_t bbmax_tool_y = -FLT_MAX;
		for (auto &it : samples) {
			bbmin_tool_x = fmin(bbmin_tool_x, it.x);
			bbmax_tool_y = fmax(bbmax_tool_y, it.y);
		}

		n_tool = samples.size();
		pos    = (float2_t*) realloc(pos,    sizeof(float2_t)*(n+n_tool));
		vel    = (float2_t*) realloc(vel,    sizeof(float2_t)*(n+n_tool));
		rho    = (float_t*)  realloc(rho,    sizeof(float_t)*(n+n_tool));
		T      = (float_t*)  realloc(T,      sizeof(float_t)*(n+n_tool));
		h      = (float_t*)  realloc(h,      sizeof(float_t)*(n+n_tool));
		fixed  = (float_t*)  realloc(fixed,  sizeof(float_t)*(n+n_tool));
		tool_p = (float_t*)  realloc(tool_p, sizeof(float_t)*(n+n_tool));

		int tool_pos_iter = 0;
		for (int i = n; i < n+n_tool; i++) {
			pos[i] = make_float2_t(samples[tool_pos_iter].x, samples[tool_pos_iter].y);
			tool_pos_iter++;

			rho[i]   = rho_tool;
			h[i]     = hdx*dx;
			vel[i].x =   0.;
			vel[i].y =   0.;
			fixed[i] =   0.;
			if (pos[i].x < bbmin_tool_x +  dx/2 || pos[i].y > bbmax_tool_y - dx/2) {
				fixed[i] = 1.;
			}
			T[i]     = joco.Tref;
			tool_p[i] = 1.;
		}
	}


	//-----------------------------------------

	//measure Bounding Box of complete domain
	float2_t bbmin = make_float2_t( FLT_MAX,  FLT_MAX);
	float2_t bbmax = make_float2_t(-FLT_MAX, -FLT_MAX);
	for (unsigned int i = 0; i < n+n_tool; i++) {
		bbmin.x = fmin(pos[i].x, bbmin.x);
		bbmin.y = fmin(pos[i].y, bbmin.y);
		bbmax.x = fmax(pos[i].x, bbmax.x);
		bbmax.y = fmax(pos[i].y, bbmax.y);
	}

	bbmin.x -= 1e-8;
	bbmin.y -= 1e-8;

	bbmax.x += 1e-8;
	bbmax.y += 1e-8;

	bbox toolbb = t->safe_bb(1e-3);
	bbmin.x = fmin(bbmin.x, toolbb.bbmin_x);
	bbmin.y = fmin(bbmin.y, toolbb.bbmin_y);
	bbmax.x = fmax(bbmax.x, toolbb.bbmax_x)+10*dx;
	bbmax.y = fmax(bbmax.y, toolbb.bbmax_y);

	float_t max_height = hi_y + hi_x - lo_x;
	bbmax.y = fmax(bbmax.y, max_height);

	//set up grid for spatial hashing
	//	- grid rothlin saves memory but is slower than grid green
	//	- each grid can be configured to adapt to the solution domain or to stay fixed
	//	- fixed grids work considerably faster than adapting ones

	//	*grid = new grid_gpu_green(10*n, n);
	*grid = new grid_gpu_green(n+n_tool, bbmin, bbmax, hdx*dx);
	//	*grid = new grid_gpu_rothlin(10*n, n);
	//	*grid = new grid_gpu_rothlin(n, bbmin, bbmax, hdx*dx);

	//check whether grid was set up correctly
	for (unsigned int i = 0; i < n + n_tool; i++) {
		bool in_x = pos[i].x > bbmin.x && pos[i].x < bbmax.x;
		bool in_y = pos[i].y > bbmin.y && pos[i].y < bbmax.y;
		if (!(in_x && in_y)) {
			printf("WARINING: particle out of inital bounding box!\n");
		}
	}

	//usui wear model
	float_t usui_K = 7.8; 	//GPa
	usui_K = 1./100*7.8;	//bomb units
	float_t usui_alpha = 2500.;
	global_wear = new tool_wear(usui_K, usui_alpha, (unsigned int) n+n_tool, phys, glm::dvec2(0., vel_tool));

	//propagate constants to actions and interactions
	actions_setup_corrector_constants(corr);
	actions_setup_physical_constants(phys);
	actions_setup_johnson_cook_constants(joco);
	actions_setup_thermal_constants_wp(trml_wp);
	actions_setup_thermal_constants_tool(trml_tool);

	interactions_setup_geometry_constants(*grid);
	interactions_setup_corrector_constants(corr);
	interactions_setup_physical_constants(phys);
	interactions_setup_thermal_constants_tool(trml_tool, global_tool);
	interactions_setup_thermal_constants_workpiece(trml_wp);

	assert(check_cuda_error());

	//CFL based choice of time step
	global_dt = 0.5*hdx*dx/(sqrt(phys.K/phys.rho0) + sqrt(vel_tool));
	global_t_final = cutting_distance / vel_tool;

	printf("timestep chosen: %e\n", global_dt);
	printf("calculating with %d regular and %d tool particles for a total of %d\n", n, n_tool, n+n_tool);

	return new particle_gpu(pos, vel, rho, T, h, fixed, tool_p, n + n_tool);
}
