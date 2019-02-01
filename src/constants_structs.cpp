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

#include "constants_structs.h"

phys_constants make_phys_constants() {
	phys_constants phys;
	memset(&phys, 0, sizeof(phys_constants));
	return phys;
}

trml_constants make_trml_constants() {
	trml_constants trml;
	memset(&trml, 0, sizeof(trml_constants));
	return trml;
}

corr_constants make_corr_constants() {
	corr_constants corr;
	memset(&corr, 0, sizeof(corr_constants));
	corr.wdeltap = 1.;
	return corr;
}

joco_constants make_joco_constants() {
	joco_constants joco;
	memset(&joco, 0, sizeof(joco_constants));
	return joco;
}

geom_constants make_geom_constants() {
	geom_constants geom;
	memset(&geom, 0, sizeof(geom_constants));
	return geom;
}
