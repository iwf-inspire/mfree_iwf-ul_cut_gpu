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

#include "geometric_primitives.h"

vec2_t line::intersect(line l) const {

	// parallel vertical lines
	if (vertical && l.vertical) {
		return vec2_t(DBL_MAX, DBL_MAX);
	}

	// parallel but not vertical lines
	if (fabs(a-l.a) < 1e-12) {
		return vec2_t(DBL_MAX, DBL_MAX);
	}

	if (vertical) {
		float_t vert_x = b;
		float_t inter_y = l.a*vert_x + l.b;
		return vec2_t(vert_x, inter_y);
	}

	if (l.vertical)  {
		float_t vert_x = l.b;
		float_t inter_y = a*vert_x + b;
		return vec2_t(vert_x, inter_y);
	}

	float_t x = (l.b-b)/(a-l.a);
	float_t y = a*x+b;
	return vec2_t(x,y);
}

vec2_t line::closest_point(vec2_t xq) const {

	if (vertical) {
		return vec2_t(b, xq.y);
	}

	float_t bb = -1;

	float_t cc = b;
	float_t aa = a;

	float_t px = (bb*( bb*xq.x - aa*xq.y) - aa*cc)/(aa*aa + bb*bb);
	float_t py = (aa*(-bb*xq.x + aa*xq.y) - bb*cc)/(aa*aa + bb*bb);

	return vec2_t(px, py);
}

line::line(float_t a, float_t b, bool vertical) : a(a), b(b), vertical(vertical) {}

line::line() {}

line::line(vec2_t p1, vec2_t p2) {
	float_t Mxx = p1.x; float_t Mxy = 1.;
	float_t Myx = p2.x; float_t Myy = 1.;

	float_t detM = Mxx*Myy - Mxy*Myx;

	if (fabs(detM) < 1e-12) {	//vertical line
		vertical = true;
		a = DBL_MAX;
		b = p1.x;	//or p2.x
		return;
	}

	a = (p1.y*Myy - Mxy*p2.y)/detM;
	b = (p2.y*Mxx - Myx*p1.y)/detM;
}

bool bbox::in(vec2_t qp) {
	bool in_x = qp.x >= bbmin_x && qp.x <= bbmax_x;
	bool in_y = qp.y >= bbmin_y && qp.y <= bbmax_y;
	return in_x && in_y;
}

bool bbox::valid() const {
	bool invalid_x = bbmax_x - bbmin_x  < 1e-12;
	bool invalid_y = bbmax_y - bbmin_y  < 1e-12;

	return !(invalid_x || invalid_y);
}

bbox::bbox() {}

bbox::bbox(vec2_t p1, vec2_t p2) {
	bbmin_x = fmin(p1.x, p2.x);
	bbmax_x = fmax(p1.x, p2.x);
	bbmin_y = fmin(p1.y, p2.y);
	bbmax_y = fmax(p1.y, p2.y);
}

bbox::bbox(float_t bbmin_x, float_t bbmax_x, float_t bbmin_y, float_t bbmax_y) :
				bbmin_x(bbmin_x), bbmax_x(bbmax_x), bbmin_y(bbmin_y), bbmax_y(bbmax_y) {}

segment::segment(vec2_t left, vec2_t right) {
	this->left  = left;
	this->right = right;

	vec2_t dist = right - left;
	vec2_t n(dist.y, -dist.x);
	n = glm::normalize(n);

	this->n = n;
	this->l = line(left,right);
}


float_t segment::length() const {
	return glm::length(left-right);
}

segment::segment() {}

circle_segment::circle_segment(float_t r, float_t t1, float_t t2, vec2_t p) : r(r), t1(t1), t2(t2), p(p) {}

circle_segment::circle_segment(vec2_t p1, vec2_t p2, vec2_t p3) {
	float_t x1 = p1.x; float_t y1 = p1.y;
	float_t x2 = p2.x; float_t y2 = p2.y;
	float_t x3 = p3.x; float_t y3 = p3.y;

	glm::dmat3x3 d1(x1*x1 + y1*y1, x2*x2 + y2*y2, x3*x3 + y3*y3, y1, y2, y3, 1. ,1., 1.);
	glm::dmat3x3 d2(x1, x2, x3, x1*x1 + y1*y1, x2*x2 + y2*y2, x3*x3 + y3*y3, 1. ,1., 1.);
	glm::dmat3x3 frac(x1, x2, x3, y1, y2, y3, 1., 1., 1.);

	p.x = glm::determinant(d1)/(2.*glm::determinant(frac));
	p.y = glm::determinant(d2)/(2.*glm::determinant(frac));

	r = glm::length(p-p1);
}

circle_segment::circle_segment() {}
