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

#include "vtk_writer.h"

void vtk_writer_write(const particle_gpu *particles, int step) {

	static int      *h_idx		= 0;
	static float2_t *h_pos		= 0;
	static float2_t *h_vel		= 0;
	static float_t  *h_rho		= 0;
	static float_t  *h_h		= 0;
	static float_t  *h_p		= 0;
	static float_t  *h_T		= 0;
	static float_t  *h_eps		= 0;

	static float4_t *h_S		= 0;

	static float_t *h_fixed		= 0;
	static float_t *h_blanked	= 0;
	static float_t *h_tool_p	= 0;

	if (h_idx == 0) {
		int n_init = particles->N;

		// Memory allocation only upon first call;
		h_idx		= new int[n_init];
		h_pos		= new float2_t[n_init];
		h_vel		= new float2_t[n_init];
		h_rho		= new float_t[n_init];
		h_h			= new float_t[n_init];
		h_p			= new float_t[n_init];
		h_T			= new float_t[n_init];
		h_eps		= new float_t[n_init];

		h_S			= new float4_t[n_init];

		h_fixed		= new float_t[n_init];		// BC-particles
		h_blanked	= new float_t[n_init];		// blanked particles
		h_tool_p	= new float_t[n_init];
	}

	int n = particles->N;

	cudaMemcpy(h_idx, particles->idx,    sizeof(int)*n,      cudaMemcpyDeviceToHost);
	cudaMemcpy(h_pos, particles->pos,    sizeof(float2_t)*n, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_vel, particles->vel,    sizeof(float2_t)*n, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_rho, particles->rho,    sizeof(float_t)*n,  cudaMemcpyDeviceToHost);
	cudaMemcpy(h_h,   particles->h,      sizeof(float_t)*n,  cudaMemcpyDeviceToHost);
	cudaMemcpy(h_p,   particles->p,      sizeof(float_t)*n,  cudaMemcpyDeviceToHost);
	cudaMemcpy(h_T,   particles->T,	 sizeof(float_t)*n,  cudaMemcpyDeviceToHost);
	cudaMemcpy(h_eps, particles->eps_pl, sizeof(float_t)*n,  cudaMemcpyDeviceToHost);

	cudaMemcpy(h_S, particles->S, sizeof(float4_t)*n,  cudaMemcpyDeviceToHost);

	cudaMemcpy(h_fixed, particles->fixed, sizeof(float_t)*n,  cudaMemcpyDeviceToHost);
	cudaMemcpy(h_tool_p, particles->tool_particle, sizeof(float_t)*n,  cudaMemcpyDeviceToHost);

	int num_unblanked_part = 0;
	for (int i = 0; i < n; i++) {
		if (h_blanked[i] != 1.) {
			num_unblanked_part++;
		}
	}

	char buf[256];
	sprintf(buf, "results/vtk_out_%06d.vtk", step);
	FILE *fp = fopen(buf, "w+");

	fprintf(fp, "# vtk DataFile Version 2.0\n");
	fprintf(fp, "mfree iwf\n");
	fprintf(fp, "ASCII\n");
	fprintf(fp, "\n");

	fprintf(fp, "DATASET UNSTRUCTURED_GRID\n");
	fprintf(fp, "POINTS %d float\n", num_unblanked_part);
	for (unsigned int i = 0; i < n; i++) {
		if (h_blanked[i]==1.) continue;
		fprintf(fp, "%f %f %f\n", h_pos[i].x, h_pos[i].y, 0.);
	}
	fprintf(fp, "\n");

	fprintf(fp, "CELLS %d %d\n", num_unblanked_part, 2*num_unblanked_part);
	for (unsigned int i = 0; i < n; i++) {
		if (h_blanked[i]==1.) continue;
		fprintf(fp, "%d %d\n", 1, i);
	}
	fprintf(fp, "\n");

	fprintf(fp, "CELL_TYPES %d\n", num_unblanked_part);
	for (unsigned int i = 0; i < n; i++) {
		if (h_blanked[i]==1.) continue;
		fprintf(fp, "%d\n", 1);
	}
	fprintf(fp, "\n");

	fprintf(fp, "POINT_DATA %d\n", num_unblanked_part);

	fprintf(fp, "SCALARS density float 1\n");
	fprintf(fp, "LOOKUP_TABLE default\n");
	for (unsigned int i = 0; i < n; i++) {
		if (h_blanked[i]==1.) continue;
		fprintf(fp, "%f\n", h_rho[i]);
	}
	fprintf(fp, "\n");

	fprintf(fp, "SCALARS Temperature float 1\n");
	fprintf(fp, "LOOKUP_TABLE default\n");
	for (unsigned int i = 0; i < n; i++) {
		if (h_blanked[i]==1.) continue;
		fprintf(fp, "%f\n", h_T[i]);
	}
	fprintf(fp, "\n");

	fprintf(fp, "SCALARS Fixed float 1\n");
	fprintf(fp, "LOOKUP_TABLE default\n");
	for (unsigned int i = 0; i < n; i++) {
		if (h_blanked[i]==1.) continue;
		fprintf(fp, "%f\n", h_fixed[i]);
	}
	fprintf(fp, "\n");

	fprintf(fp, "SCALARS Tool float 1\n");
	fprintf(fp, "LOOKUP_TABLE default\n");
	for (unsigned int i = 0; i < n; i++) {
		if (h_blanked[i]==1.) continue;
		fprintf(fp, "%f\n", h_tool_p[i]);
	}
	fprintf(fp, "\n");

	fprintf(fp, "SCALARS EquivAccumPlasticStrain float 1\n");
	fprintf(fp, "LOOKUP_TABLE default\n");
	for (unsigned int i = 0; i < n; i++) {
		if (h_blanked[i]==1.) continue;
		fprintf(fp, "%f\n", h_eps[i]);
	}
	fprintf(fp, "\n");

	fprintf(fp, "VECTORS Velocity float\n");
	for (unsigned int i = 0; i < n; i++) {
		if (h_blanked[i]==1.) continue;
		fprintf(fp, "%f %f %f\n", h_vel[i].x, h_vel[i].y, 0.);
	}
	fprintf(fp, "\n");

	fprintf(fp, "SCALARS SvM float 1\n");
	fprintf(fp, "LOOKUP_TABLE default\n");
	for (unsigned int i = 0; i < n; i++) {
		if (h_blanked[i]==1.) continue;

		float_t cxx = h_S[i].x - h_p[i];
		float_t cyy = h_S[i].z - h_p[i];
		float_t czz = h_S[i].w - h_p[i];
		float_t cxy = h_S[i].y;

		float_t svm2 =  (cxx*cxx + cyy*cyy + czz*czz) - cxx * cyy - cxx * czz - cyy * czz + 3.0 * cxy * cxy;

		float_t svm = (svm2 > 0) ? sqrt(svm2) : 0.;
		fprintf(fp, "%f\n", svm);
	}
	fprintf(fp, "\n");

	fclose(fp);
}

struct triangle {
	vec2_t p1,p2,p3;
	triangle(vec2_t p1, vec2_t p2, vec2_t p3) : p1(p1), p2(p2), p3(p3) {}
};

void vtk_writer_write(const tool* tool, int step) {
	auto segments = tool->get_segments();
	assert(segments.size() == 4 || segments.size() == 5);

	std::vector<triangle> triangles;

	//mesh tool "body"
	if (segments.size() == 4) {
        triangles.push_back(triangle(segments[0].left, segments[0].right, segments[1].right));
        triangles.push_back(triangle(segments[2].left, segments[2].right, segments[3].right));
	} else if (segments.size() == 5) {
        triangles.push_back(triangle(segments[0].left, segments[0].right, segments[2].right));
        triangles.push_back(triangle(segments[1].left, segments[1].right, segments[2].right));
        triangles.push_back(triangle(segments[3].left, segments[3].right, segments[4].right));
	}

	//mesh fillet
	if (tool->get_fillet() != 0) {
		const int num_discr = 20;
		auto fillet = tool->get_fillet();
		float_t t1 = fmin(fillet->t1, fillet->t2);
		float_t t2 = fmax(fillet->t1, fillet->t2);

		float_t lo = t1 - 0.1*t1;
		float_t hi = t2 + 0.1*t2;

		float_t d_angle = (t2-t1)/(num_discr-1);

		float_t r = fillet->r;

		for (int i = 0; i < num_discr-1; i++) {
			float_t angle_1 = lo + (i+0)*d_angle;
			float_t angle_2 = lo + (i+1)*d_angle;

			vec2_t p1 = vec2_t(fillet->p.x, fillet->p.y);
			vec2_t p2 = vec2_t(p1.x + r*sin(angle_1), p1.y + r*cos(angle_1));
			vec2_t p3 = vec2_t(p1.x + r*sin(angle_2), p1.y + r*cos(angle_2));
			triangles.push_back(triangle(p1, p2, p3));
		}
	}

	int num_tri = triangles.size();

	char buf[256];
	sprintf(buf, "results/vtk_tool_%06d.vtk", step);
	FILE *fp = fopen(buf, "w+");

	fprintf(fp, "# vtk DataFile Version 2.0\n");
	fprintf(fp, "mfree iwf\n");
	fprintf(fp, "ASCII\n");
	fprintf(fp, "\n");
	fprintf(fp, "DATASET UNSTRUCTURED_GRID\n");
	fprintf(fp, "POINTS %d float\n", 3*num_tri);

	for (auto it : triangles) {
		fprintf(fp, "%f %f %f\n", it.p1.x, it.p1.y, 0.);
		fprintf(fp, "%f %f %f\n", it.p2.x, it.p2.y, 0.);
		fprintf(fp, "%f %f %f\n", it.p3.x, it.p3.y, 0.);
	}
	fprintf(fp, "\n");

	fprintf(fp, "CELLS %d %d\n", num_tri, 3*num_tri + num_tri);
	for (int i = 0; i < num_tri; i++) {
		fprintf(fp, "3 %d %d %d\n", 3*i+0, 3*i+1, 3*i+2);
	}
	fprintf(fp, "\n");

	fprintf(fp, "CELL_TYPES %d\n", num_tri);
	for (int i = 0; i < num_tri; i++) {
		fprintf(fp, "5\n");
	}

	fclose(fp);

}
