
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>


__global__ void print_values() {

	printf("ThreadIdx.x: %d ThreadIdx.y: %d ThreadIdx.z: %d\t BlockIdx.x: %d BlockIdx.y: %d BlockIdx.z: %d\t gridDim.x: %d gridDim.y: %d gridDim.z: %d \n", threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z, gridDim.x, gridDim.y, gridDim.z);
}

int main() {

	int nx = 4;
	int ny = 4;
	int nz = 4;

	dim3 block(2, 2, 2);
	dim3 grid(nx / block.x, ny / block.y, nz / block.z);

	print_values <<<grid, block>>> ();

	cudaDeviceSynchronize();
	cudaDeviceReset();

	return 0;
}