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

#include "tool.h"

static vec2_t solve_quad(float_t a, float_t b, float_t c) {
	float_t x1 = (-b + sqrt(b*b-4*a*c))/(2*a);
	float_t x2 = (-b - sqrt(b*b-4*a*c))/(2*a);

	return vec2_t(x1,x2);
}

static float_t myatan2(float_t y, float_t x) {
	float_t t = atan2(y,x);
	if (t > 0.) {
		return t;
	} else {
		return t + 2*M_PI;
	}
}

vec2_t tool::fit_fillet(float_t r, line lm, line l1) const {

	if (l1.vertical) {
//		vec2_t br = lm.intersect(l1);
//		float_t y = br.y + r;
//		float_t x = (y-lm.b)/lm.a;
//		return vec2_t(x,y);

		line lparallel = line(DBL_MAX, l1.b - r, true);
		return lparallel.intersect(lm);
	}

	float_t A0 = lm.a;
	float_t B0 = lm.b;

	float_t a = l1.a;
	float_t b = l1.b;

	float_t A = a-A0;
	float_t B = b-B0;
	float_t C = r*sqrt(a*a+1.);

	vec2_t sol = solve_quad(A*A, 2*A*B, B*B-C*C);
	float_t xm = fmin(sol.x, sol.y);
	float_t ym = lm.a*xm + lm.b;
	return vec2_t(xm, ym);
}

void tool::construct_segments(std::vector<vec2_t> list_p) {
	unsigned int n = list_p.size();
	for (unsigned int i = 0; i < n; i++) {
		unsigned int cur  = i;
		unsigned int next = (cur+1 > n-1) ? 0 : i+1;
		m_segments.push_back(segment(list_p[cur], list_p[next]));
	}
}

std::vector<vec2_t> tool::construct_points_and_fillet(vec2_t tl, vec2_t tr, vec2_t br, vec2_t bl, float_t r) {
	// construct line halfing the space between l1, l2 => lm
	vec2_t pm = br;
	vec2_t nt = tr - br;
	vec2_t nl = bl - br;

	nt = glm::normalize(nt);
	nl = glm::normalize(nl);

	vec2_t nm = float_t(0.5)*(nt+nl);
	nm = glm::normalize(nm);

	line lm(pm, pm+nm);

	// find center of fillet => p
	line l1(tr, br);
	line l2(bl, br);
	vec2_t p = fit_fillet(r, lm, l1); // fit_fillet(r, lm, l2) would work too

	// find points on l1, l2 that meet the fillet => trc, blc (c = "continued")
	vec2_t trc = l1.closest_point(p);
	vec2_t blc = l2.closest_point(p);

	// construct circle segment
	float_t t1 = myatan2(p.y - trc.y, p.x - trc.x);
	float_t t2 = myatan2(p.y - blc.y, p.x - blc.x);
	m_fillet = new circle_segment(r, t1, t2, p);

	return std::vector<vec2_t>({tl, tr, trc, blc, bl});
}

bool tool::inside(float2_t pos) const {
	vec2_t qp(pos.x, pos.y);

	bool in = true;
	for (int i = 0; i < m_segments.size(); i++) {
		vec2_t left(m_segments[i].left.x, m_segments[i].left.y);
		vec2_t n(m_segments[i].n.x, m_segments[i].n.y);
		in = in && glm::dot(left - qp, n) < 0.;
	}

	if (m_fillet) {
		in = in || glm::length(m_fillet->p - qp) < m_fillet->r;
	}

	return in;
}

std::vector<float2_t> tool::sample_tool(float_t dx, float_t T_init, float_t rho, float_t hdx) {
	std::vector<float2_t> samples;

	bbox box = safe_bb(0.1*dx);

	int nx = (box.bbmax_x - box.bbmin_x)/dx+1;
	int ny = (box.bbmax_y - box.bbmin_y)/dx+1;

	for (int i = 0; i < nx; i++) {
		for (int j = 0; j < ny; j++) {
			float2_t pos;
			pos.x = i*dx + box.bbmin_x;
			pos.y = j*dx + box.bbmin_y;

			if (inside(pos)) {
				samples.push_back(pos);
			}
		}
	}

	assert(samples.size() > 0);

	return samples;
}

void add_thermal_boundary(std::vector<unsigned int> idx) {

}

void tool::get_chamfer_data(vec2_t &p, float_t &r) const {
	p.x = m_fillet->p.x;
	p.y = m_fillet->p.y;
	r   = m_fillet->r;
}

bbox tool::safe_bb(float_t safety) const {

	float_t x_min = DBL_MAX; float_t x_max = -DBL_MAX;
	float_t y_min = DBL_MAX; float_t y_max = -DBL_MAX;

	for (auto it = m_segments.begin(); it != m_segments.end(); ++it) {
		x_min = fmin(it->left.x, x_min);
		x_max = fmax(it->left.x, x_max);
		y_min = fmin(it->left.y, y_min);
		y_max = fmax(it->left.y, y_max);
	}

	if (m_fillet) {
		y_min = low();
		x_max = m_fillet->p.x + m_fillet->r;
	}

	x_min -= safety;
	y_min -= safety;
	x_max += safety;
	y_max += safety;

	bbox ret;
	ret.bbmin_x = x_min;
	ret.bbmax_x = x_max;
	ret.bbmin_y = y_min;
	ret.bbmax_y = y_max;

	return ret;
}

float_t tool::low() const {
	if (m_fillet) {
		return m_fillet->p.y - m_fillet->r;
	}

	float_t y_min = DBL_MAX;
	for (auto it = m_segments.begin(); it != m_segments.end(); ++it) {
		y_min = fmin(y_min, it->left.y);
	}
	return y_min;
}

float_t tool::front() const {
	return m_front;
}

circle_segment *tool::get_fillet() const {
	return m_fillet;
}

std::vector<segment> tool::get_segments() const {
	return m_segments;
}

void tool::update_tool(float_t dt) {
	for (auto it = m_segments.begin(); it != m_segments.end(); ++it) {
		it->left  += dt*m_velocity;
		it->right += dt*m_velocity;
		it->l = line(it->left, it->right);
	}

	if (m_fillet) {
		m_fillet->p += dt*m_velocity;
	}
}


void tool::set_vel(vec2_t vel) {
	m_velocity = vel;
}

void tool::set_thermal(bool thermal) {
	m_thermal = thermal;
}

bool tool::thermal() const {
	return m_thermal;
}

vec2_t tool::get_vel() const{
	return m_velocity;
}

vec2_t tool::center() const {
	float_t cx = 0.;
	float_t cy = 0.;

	for (auto it = m_segments.begin(); it != m_segments.end(); ++it) {
		cx += it->left.x;
		cy += it->left.y;
	}

	cx /= m_segments.size();
	cy /= m_segments.size();

	return vec2_t(cx, cy);
}

void tool::print(FILE *fp) {

	fprintf(fp, "%lu %lu\n", m_segments.size(), m_segments.size());

	for (auto it = m_segments.begin(); it != m_segments.end(); ++it) {
		fprintf(fp, "%f %f\n", it->left.x, it->left.y);
	}
	fprintf(fp, "%f %f\n", m_segments.back().right.x, m_segments.back().right.y);

	if (m_fillet) {
		fprintf(fp, "%f %f\n", m_fillet->p.x, m_fillet->p.y);
		fprintf(fp, "%f %f\n", m_fillet->r, m_fillet->r);
	} else {
		fprintf(fp, "0 0\n");
		fprintf(fp, "0 0\n");
	}

}

void tool::print() {
	printf("%lu %lu\n", m_segments.size(), m_segments.size());

	for (auto it = m_segments.begin(); it != m_segments.end(); ++it) {
		printf("%f %f\n", it->left.x, it->left.y);
	}
	printf("%f %f\n", m_segments.back().right.x, m_segments.back().right.y);

	if (m_fillet) {
		printf("%f %f\n", m_fillet->p.x, m_fillet->p.y);
		printf("%f %f\n", m_fillet->r, m_fillet->r);
	} else {
		printf("0 0\n");
		printf("0 0\n");
	}
}

void tool::print(unsigned int step, const char *folder_name) {
	char buf[256];
	sprintf(buf, "./%s/tool_%07d.txt", folder_name, step);

	FILE *fp = fopen(buf, "w+");
	if (!fp) {
		printf("could not open tool file\n!");
		exit(-1);
	}

	fprintf(fp, "%d\n", (int) m_segments.size());
	for (auto it = m_segments.begin(); it != m_segments.end(); ++it) {
		fprintf(fp, "%f %f\n", it->left.x, it->left.y);
	}

	if (m_fillet) {
		fprintf(fp, "%f %f %f %f %f\n", m_fillet->p.x, m_fillet->p.y, m_fillet->r, m_fillet->t1, m_fillet->t2);
	}

	fclose(fp);
}

float_t tool::mu() const {
	return m_mu;
}

void tool::set_chamfer(vec2_t cp, float_t r, float_t t1, float_t t2) {
	m_fillet = new circle_segment(r, t1, t2, cp);
}

void tool::set_chamfer_debug(bool chamfer_debug) {
	m_chamfer_debug = chamfer_debug;
}

tool::tool(vec2_t tl, vec2_t tr, vec2_t br, vec2_t bl, float_t mu_fric) : m_mu(mu_fric) {
	construct_segments(std::vector<vec2_t>({tl, tr, br, bl}));
	m_front = br.x;
}

tool::tool(vec2_t tl, vec2_t tr, vec2_t br, vec2_t bl, float_t r, float_t mu_fric) : m_mu(mu_fric)  {
	if (r == 0.) {
		construct_segments(std::vector<vec2_t>({tl, tr, br, bl}));
		m_front = br.x;
		return;
	}

	std::vector<vec2_t> points = construct_points_and_fillet(tl,tr,br,bl,r);
	construct_segments(points);
	m_front = br.x;
}

tool::tool(vec2_t tl, float_t length, float_t height,
		float_t rake_angle, float_t clearance_angle,
		float_t r, float_t mu_fric) : m_mu(mu_fric)  {

	vec2_t tr(tl.x+length, tl.y);
	vec2_t bl(tl.x, tl.y-height);

	float_t alpha_rake = rake_angle * M_PI / 180.;
	float_t alpha_free = (180-90-clearance_angle) * M_PI / 180.;

	mat2x2_t rot_rake(cos(alpha_rake), -sin(alpha_rake), sin(alpha_rake), cos(alpha_rake));
	mat2x2_t rot_free(cos(alpha_free), -sin(alpha_free), sin(alpha_free), cos(alpha_free));

	vec2_t down(0., -1.);

	vec2_t trc = tr + down*rot_rake;
	vec2_t blc = bl + down*rot_free;

	line l1(tr,trc);
	line l2(bl,blc);

	vec2_t br = l1.intersect(l2);

	if (r == 0.) {
		construct_segments(std::vector<vec2_t>({tl, tr, br, bl}));
		m_front = br.x;
		return;
	}

	std::vector<vec2_t> points = construct_points_and_fillet(tl,tr,br,bl,r);
	construct_segments(points);

	m_front = br.x;
}

tool::tool(vec2_t tl, float_t length, float_t height,
		float_t rake_angle, float_t clearance_angle,
		float_t mu_fric) : m_mu(mu_fric)  {
	//		// returns distance to circle segment if closest
	//		// point falls between t1, t2
	//		// return DBL_MAX otherwise
	//		float_t distance(vec2_t qp) const;
	//
	//		// intersect circle segment with line segment p1,p2
	//		// returns: 0 if no intersection point falls between t1, t2 or p1,p2 misses segment completely
	//		//          1 if one intersection point falls between t1, t2. i1 is set
	//		//          2 if both intersection points fall between t1, t2. i1, i2 is set
	//		unsigned int intersect(vec2_t p1, vec2_t p2, vec2_t &i1, vec2_t &i2);
	vec2_t tr(tl.x+length, tl.y);
	vec2_t bl(tl.x, tl.y-height);

	float_t alpha_rake = rake_angle * M_PI / 180.;
	float_t alpha_free = (180-90-clearance_angle) * M_PI / 180.;

	mat2x2_t rot_rake(cos(alpha_rake), -sin(alpha_rake), sin(alpha_rake), cos(alpha_rake));
	mat2x2_t rot_free(cos(alpha_free), -sin(alpha_free), sin(alpha_free), cos(alpha_free));

	vec2_t down(0., -1.);

	vec2_t trc = tr + down*rot_rake;
	vec2_t blc = bl + down*rot_free;

	line l1(tr,trc);
	line l2(bl,blc);

	vec2_t br = l1.intersect(l2);

	construct_segments(std::vector<vec2_t>({tl, tr, br, bl}));

	m_front = br.x;
}

tool::tool() {}
