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

#include "tool_gpu.cuh"
#include "tool_gpu.h"

__constant__ segment_gpu segments[TOOL_MAX_SEG];
__constant__ int num_seg;
__constant__ circle_segment_gpu fillet;
__constant__ float_t friction_mu;
__constant__ float_t contact_alpha;
__constant__ float_t slave_mass;
__constant__ float2_t tool_vel;

__device__ float_t myatan2(float_t y, float_t x) {
	double t = atan2(y,x);
	if (t > 0.) {
		return t;
	} else {
		return t + 2*M_PI;
	}
}

__device__ float_t circle_segment_distance(circle_segment_gpu fillet, vec2_t qp) {
	vec2_t  p  = vec2_t(fillet.p.x, fillet.p.y);
	float_t r  = fillet.r;
	float_t t1 = fillet.t1;
	float_t t2 = fillet.t2;

	float_t Ax = p.x;
	float_t Ay = p.y;

	float_t Bx = qp.x;
	float_t By = qp.y;

	float_t Cx = Ax + r*(Bx-Ax)/sqrt((Bx-Ax)*(Bx-Ax)+(By-Ay)*(By-Ay));
	float_t Cy = Ay + r*(By-Ay)/sqrt((Bx-Ax)*(Bx-Ax)+(By-Ay)*(By-Ay));

	vec2_t cp(Cx, Cy);
	float_t t = myatan2(p.y-cp.y, p.x-cp.x);

	bool valid = t > fmin(t1,t2) && t < fmax(t1,t2);

	if (valid) {
		return glm::length(cp-qp);
	} else {
		return FLT_MAX;
	}
}

__device__ vec2_t segment_closest_point(line_gpu l, vec2_t xq) {
	if (l.vertical) {
		return vec2_t(l.b, xq.y);
	}

	float_t b = l.b;
	float_t a = l.a;

	float_t bb = -1;
	float_t cc = b;
	float_t aa = a;

	float_t px = (bb*( bb*xq.x - aa*xq.y) - aa*cc)/(aa*aa + bb*bb);
	float_t py = (aa*(-bb*xq.x + aa*xq.y) - bb*cc)/(aa*aa + bb*bb);

	return vec2_t(px, py);
}

// contact to establish contact between the tool and a particle at location qp (query point)
// fills closest point (cp) and surface normal (n)
//		algorithm simply checks each geometrical primitive for penetration
//		see segment_closest_point and circle_segment_distance
__device__ bool establish_contact(vec2_t qp, vec2_t &cp, vec2_t &n) {
	bool has_fillet = num_seg == TOOL_MAX_SEG;

	bool in = true;
	for (int i = 0; i < num_seg; i++) {
		vec2_t left(segments[i].left.x, segments[i].left.y);
		vec2_t n(segments[i].n.x, segments[i].n.y);
		in = in && glm::dot(left - qp, n) < 0.;
	}

	if (has_fillet) {
		vec2_t p(fillet.p.x, fillet.p.y);
		in = in || glm::length(p - qp) < fillet.r;
	}

	if (!in) return false;

	float_t depth = FLT_MAX;

	for (int i = 0; i < num_seg; i++) {

		float_t d;
		if (segments[i].l.vertical) {
			d = fabs(segments[i].left.x - qp.x);		//right would work too
			if (d < depth) {
				depth = d;

				n  = glm::normalize(vec2_t(segments[i].left.x - qp.x, 0.));
				cp = vec2_t(segments[i].left.x, qp.y);
			}
		} else if (has_fillet && i==2) {					//if there is a fillet present, its always covering the third segment
			d = circle_segment_distance(fillet, qp);
			if (d < depth) {
				depth = d;
				vec2_t p(fillet.p.x, fillet.p.y);

				n  = glm::normalize(qp - p);
				cp = p + n*fillet.r;
			}
		} else {
			vec2_t p = segment_closest_point(segments[i].l, qp);
			d = glm::length(p-qp);
			if (d < depth) {
				depth = d;

				n  = vec2_t(-segments[i].n.x, -segments[i].n.y);
				cp = p;
			}
		}
	}

	return true;
}

__device__ vec2_t compute_friction_force(vec2_t fN, vec2_t vr, float_t mu, float_t ms, float_t dt) {
	vec2_t ft = -ms/dt*vr;
	vec2_t fT = -fmin(mu*glm::length(fN), glm::length(ft))*glm::normalize(vr);
	return fT;
}

//kernel that computs contact forces
//	1.) establish contact / measure penetration
//	2.) apply a penalty force proportional to penetration (force term according to suggestion by nianfei 2009)
//	3.) friction force according to coulomb friction with stabilizations suggested in the LSDYNA theory manual
__global__ void do_compute_contact_forces(particle_gpu particles, float_t dt) {
	unsigned int pidx = blockIdx.x * blockDim.x + threadIdx.x;
	if (pidx >= particles.N) return;
	if (particles.tool_particle[pidx]) return;

	float2_t pos = particles.pos[pidx];
	float2_t vel = particles.vel[pidx];

	vec2_t xs(pos.x, pos.y);
	vec2_t vs(vel.x, vel.y);

	vec2_t xm;
	vec2_t n;

	bool inside = establish_contact(xs, xm, n);
	if (!inside) {
		particles.fc[pidx].x = 0.;
		particles.fc[pidx].y = 0.;
		particles.ft[pidx].x = 0.;
		particles.ft[pidx].y = 0.;
		particles.n[pidx].x  = 0.;
		particles.n[pidx].y  = 0.;
		return;
	}

	float_t dt2 = dt*dt;
	float_t gN  = glm::dot((xs-xm),n);
	vec2_t  fN  = -slave_mass*gN*n/dt2*contact_alpha;	//nianfei 2009
	vec2_t  fT(0.,0.);

	if (friction_mu != 0.) {
		vec2_t vm = vec2_t(tool_vel.x, tool_vel.y);
		vec2_t v = vs-vm;
		vec2_t vr = v - v*n;	//relative velocity

		//---- lsdyna theory manual ----
		glm::dvec2 fricold(particles.ft[pidx].x, particles.ft[pidx].y);

		glm::dvec2 kdeltae = contact_alpha*slave_mass*vr/dt;
		double fy = friction_mu*glm::length(fN);	//coulomb friction
		glm::dvec2 fstar = fricold - kdeltae;

		if (glm::length(fstar) > fy) {
			fT  = fy*fstar/glm::length(fstar);
		} else {
			fT = fstar;
		}
		//-----------------------
	}

	particles.fc[pidx].x = fN.x;
	particles.fc[pidx].y = fN.y;
	particles.ft[pidx].x = fT.x;
	particles.ft[pidx].y = fT.y;
	particles.n[pidx].x  = n.x;
	particles.n[pidx].y  = n.y;
}

//move tool particles with tool velocity
__global__ void update_tool_particles(particle_gpu particles, float2_t tool_vel, float_t dt) {
	unsigned int pidx = blockIdx.x * blockDim.x + threadIdx.x;
	if (pidx >= particles.N) return;
	if (particles.tool_particle[pidx] == 0.) return;

	float2_t pos = particles.pos[pidx];

	pos.x += dt*tool_vel.x;
	pos.y += dt*tool_vel.y;

	particles.pos[pidx] = pos;
}

//copy cpu representation of tool to gpu memory
void tool_gpu_set_up_tool(tool *tool, float_t alpha, phys_constants phys) {
	segment_gpu segments_gpu[TOOL_MAX_SEG];
	std::vector<segment> segments_cpu = tool->get_segments();
	int num_segments_cpu = segments_cpu.size();

	for (int i = 0; i < num_segments_cpu; i++) {
		segments_gpu[i].left.x = segments_cpu[i].left.x;
		segments_gpu[i].left.y = segments_cpu[i].left.y;

		segments_gpu[i].right.x = segments_cpu[i].right.x;
		segments_gpu[i].right.y = segments_cpu[i].right.y;

		segments_gpu[i].n.x = segments_cpu[i].n.x;
		segments_gpu[i].n.y = segments_cpu[i].n.y;

		segments_gpu[i].l.a = segments_cpu[i].l.a;
		segments_gpu[i].l.b = segments_cpu[i].l.b;
		segments_gpu[i].l.vertical = segments_cpu[i].l.vertical;
	}

	circle_segment_gpu fillet_gpu;
	circle_segment *fillet_cpu = tool->get_fillet();

	if (fillet_cpu) {
		fillet_gpu.p.x = fillet_cpu->p.x;
		fillet_gpu.p.y = fillet_cpu->p.y;
		fillet_gpu.r   = fillet_cpu->r;
		fillet_gpu.t1  = fillet_cpu->t1;
		fillet_gpu.t2  = fillet_cpu->t2;
	}

	glm::dvec2 tool_vel_cpu = tool->get_vel();
	float2_t tool_vel_gpu;
	tool_vel_gpu.x = tool_vel_cpu.x;
	tool_vel_gpu.y = tool_vel_cpu.y;

	float_t mu_cpu = tool->mu();

	cudaMemcpyToSymbol(segments, segments_gpu, sizeof(segment_gpu)*TOOL_MAX_SEG, 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(num_seg,  &num_segments_cpu, sizeof(int), 0, cudaMemcpyHostToDevice);
	if (fillet_cpu) {
		cudaMemcpyToSymbol(fillet, &fillet_gpu, sizeof(circle_segment_gpu), 0, cudaMemcpyHostToDevice);
	}
	cudaMemcpyToSymbol(friction_mu, &mu_cpu, sizeof(float_t), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(contact_alpha, &alpha, sizeof(float_t), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(slave_mass, &phys.mass, sizeof(float_t), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(tool_vel, &tool_vel_gpu, sizeof(float2_t), 0, cudaMemcpyHostToDevice);
}

//move segments on GPU AND CPU with tool velocity
//	CPU segments are only updated for visualization purposes
void tool_gpu_update_tool(tool *tool, particle_gpu *particles) {
	tool->update_tool(global_dt);

	segment_gpu segments_gpu[TOOL_MAX_SEG];
	std::vector<segment> segments_cpu = tool->get_segments();
	int num_segments_cpu = segments_cpu.size();

	for (int i = 0; i < num_segments_cpu; i++) {
		segments_gpu[i].left.x = segments_cpu[i].left.x;
		segments_gpu[i].left.y = segments_cpu[i].left.y;

		segments_gpu[i].right.x = segments_cpu[i].right.x;
		segments_gpu[i].right.y = segments_cpu[i].right.y;

		segments_gpu[i].n.x = segments_cpu[i].n.x;
		segments_gpu[i].n.y = segments_cpu[i].n.y;

		segments_gpu[i].l.a = segments_cpu[i].l.a;
		segments_gpu[i].l.b = segments_cpu[i].l.b;
		segments_gpu[i].l.vertical = segments_cpu[i].l.vertical;
	}
	cudaMemcpyToSymbol(segments, segments_gpu, sizeof(segment_gpu)*TOOL_MAX_SEG, 0, cudaMemcpyHostToDevice);


	circle_segment_gpu fillet_gpu;
	circle_segment *fillet_cpu = tool->get_fillet();

	if (fillet_cpu) {
		fillet_gpu.p.x = fillet_cpu->p.x;
		fillet_gpu.p.y = fillet_cpu->p.y;
		fillet_gpu.r   = fillet_cpu->r;
		fillet_gpu.t1  = fillet_cpu->t1;
		fillet_gpu.t2  = fillet_cpu->t2;

		cudaMemcpyToSymbol(fillet, &fillet_gpu, sizeof(circle_segment_gpu), 0, cudaMemcpyHostToDevice);
	}

	dim3 dG((particles->N + BLOCK_SIZE-1) / BLOCK_SIZE);
	dim3 dB(BLOCK_SIZE);
	vec2_t vel = tool->get_vel();
	update_tool_particles<<<dG,dB>>>(*particles, make_float2_t(vel.x, vel.y), global_dt);
}

//entry point for computation of contact forces
void compute_contact_forces(particle_gpu *particles) {
	dim3 dG((particles->N + BLOCK_SIZE-1) / BLOCK_SIZE);
	dim3 dB(BLOCK_SIZE);
	do_compute_contact_forces<<<dG,dB>>>(*particles, global_dt);
	cudaThreadSynchronize();
}
