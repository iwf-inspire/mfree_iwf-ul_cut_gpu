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

#include "types.h"

bool check_cuda_error() {
	cudaDeviceSynchronize();

	// check for error
	cudaError_t error = cudaGetLastError();
	if(error != cudaSuccess) {
		// print the CUDA error message and exit
		printf("CUDA error: %s\n", cudaGetErrorString(error));
		return false;
	}

	return true;
}

bool check_cuda_error(const char *marker) {
	cudaDeviceSynchronize();

	// check for error
	cudaError_t error = cudaGetLastError();
	if(error != cudaSuccess) {
		// print the CUDA error message and exit
		printf("%s: CUDA error: %s\n", marker, cudaGetErrorString(error));
		return false;
	}

	return true;
}
