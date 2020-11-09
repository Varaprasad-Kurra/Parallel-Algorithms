#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include<opencv2/opencv.hpp>
#include<iostream>
using namespace std;
using namespace cv;


//__global__ void print_values() {

	//printf("ThreadIdx.x: %d ThreadIdx.y: %d ThreadIdx.z: %d\t BlockIdx.x: %d BlockIdx.y: %d BlockIdx.z: %d\t gridDim.x: %d gridDim.y: %d gridDim.z: %d \n", threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z, gridDim.x, gridDim.y, gridDim.z);
//}

int main() {

	FILE *input;
	char get_char;
	input = fopen("download.jfif", "rb");
	while ((get_char = fgetc(input)) != EOF)
	{
		printf("%c ", get_char);
	}
	fclose(input);

	

	//print_values << <grid, block >> > ();

	//cudaDeviceSynchronize();
	//cudaDeviceReset();

	return 0;
}