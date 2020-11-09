#include <chrono>
#include <cstring>
#include <iostream>

#include "CUDALERP.cuh"

#define VC_EXTRALEAN
#define WIN32_LEAN_AND_MEAN

using namespace std::chrono;

__global__ void
#ifndef __INTELLISENSE__
__launch_bounds__(256, 0)
#endif
CUDANERP_kernel(const cudaTextureObject_t d_img_tex, const float gxs, const float gys, uint8_t* __restrict const d_out, const int neww) {
	uint32_t x = (blockIdx.x << 9) + (threadIdx.x << 1);
	const uint32_t y = blockIdx.y;
	const float fy = y * gys;
#pragma unroll
	for (int i = 0; i < 2; ++i, ++x) {
		const float fx = x * gxs;
		float res = 255.0f*tex2D<float>(d_img_tex, fx, fy);
		if (x < neww) d_out[y*neww + x] = res;
	}
}

__global__ void
#ifndef __INTELLISENSE__
__launch_bounds__(256, 0)
#endif
CUDALERP_kernel(const cudaTextureObject_t d_img_tex, const float gxs, const float gys, uint8_t* __restrict const d_out, const int neww) {
	uint32_t x = (blockIdx.x << 9) + (threadIdx.x << 1);
	const uint32_t y = blockIdx.y;
	const float fy = (y + 0.5f)*gys - 0.5f;
	const float wt_y = fy - floor(fy);
	const float invwt_y = 1.0f - wt_y;
#pragma unroll
	for (int i = 0; i < 2; ++i, ++x) {
		const float fx = (x + 0.5f)*gxs - 0.5f;
		// less accurate and not really much (or any) faster
		// -----------------
		// const float res = tex2D<float>(d_img_tex, fx, fy);
		// -----------------
		const float4 f = tex2Dgather<float4>(d_img_tex, fx + 0.5f, fy + 0.5f);
		const float wt_x = fx - floor(fx);
		const float invwt_x = 1.0f - wt_x;
		const float xa = invwt_x * f.w + wt_x * f.z;
		const float xb = invwt_x * f.x + wt_x * f.y;
		const float res = 255.0f*(invwt_y*xa + wt_y * xb) + 0.5f;
		// -----------------
		if (x < neww) d_out[y*neww + x] = res;
	}
}

void CUDANERP(const cudaTextureObject_t d_img_tex, const int oldw, const int oldh, uint8_t* __restrict const d_out, const uint32_t neww, const uint32_t newh) {
	const float gxs = static_cast<float>(oldw) / static_cast<float>(neww);
	const float gys = static_cast<float>(oldh) / static_cast<float>(newh);
	CUDANERP_kernel << < {((neww - 1) >> 9) + 1, newh}, 256 >> > (d_img_tex, gxs, gys, d_out, neww);
	cudaDeviceSynchronize();
}

void CUDALERP(const cudaTextureObject_t d_img_tex, const int oldw, const int oldh, uint8_t* __restrict const d_out, const uint32_t neww, const uint32_t newh) {
	const float gxs = static_cast<float>(oldw) / static_cast<float>(neww);
	const float gys = static_cast<float>(oldh) / static_cast<float>(newh);
	CUDALERP_kernel << < {((neww - 1) >> 9) + 1, newh}, 256 >> > (d_img_tex, gxs, gys, d_out, neww);
	cudaDeviceSynchronize();
}

int main() {
	constexpr auto warmups = 2000;
	constexpr auto runs = 2000;

	auto image = new uint8_t[4];
	image[0] = 255;
	image[1] = 255;
	image[2] = 0;
	image[3] = 0;

	constexpr int oldw = 2;
	constexpr int oldh = 2;
	constexpr int neww = static_cast<int>(static_cast<double>(oldw) * 400.0);
	constexpr int newh = static_cast<int>(static_cast<double>(oldh) * 1000.0);
	const size_t total = static_cast<size_t>(neww)*static_cast<size_t>(newh);

	// ------------- CUDALERP ------------

	// setting cache and shared modes
	cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
	cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeFourByte);

	// allocating and transferring image and binding to texture object
	cudaChannelFormatDesc chandesc_img = cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindUnsigned);
	cudaArray* d_img_arr;
	cudaMallocArray(&d_img_arr, &chandesc_img, oldw, oldh, cudaArrayTextureGather);
	cudaMemcpyToArray(d_img_arr, 0, 0, image, oldh * oldw, cudaMemcpyHostToDevice);
	struct cudaResourceDesc resdesc_img;
	memset(&resdesc_img, 0, sizeof(resdesc_img));
	resdesc_img.resType = cudaResourceTypeArray;
	resdesc_img.res.array.array = d_img_arr;
	struct cudaTextureDesc texdesc_img;
	memset(&texdesc_img, 0, sizeof(texdesc_img));
	texdesc_img.addressMode[0] = cudaAddressModeClamp;
	texdesc_img.addressMode[1] = cudaAddressModeClamp;
	texdesc_img.readMode = cudaReadModeNormalizedFloat;
	texdesc_img.filterMode = cudaFilterModePoint;
	texdesc_img.normalizedCoords = 0;
	cudaTextureObject_t d_img_tex = 0;
	cudaCreateTextureObject(&d_img_tex, &resdesc_img, &texdesc_img, nullptr);

	uint8_t* d_out = nullptr;
	cudaMalloc(&d_out, total);

	for (int i = 0; i < warmups; ++i) CUDALERP(d_img_tex, oldw, oldh, d_out, neww, newh);
	auto start = high_resolution_clock::now();
	for (int i = 0; i < runs; ++i) CUDALERP(d_img_tex, oldw, oldh, d_out, neww, newh);
	auto end = high_resolution_clock::now();
	auto sum = (end - start) / runs;

	auto h_out = new uint8_t[neww * newh];
	cudaMemcpy(h_out, d_out, total, cudaMemcpyDeviceToHost);

	std::cout << "CUDA reports " << cudaGetErrorString(cudaGetLastError()) << std::endl;

	std::cout << "CUDALERP took " << static_cast<double>(sum.count()) * 1e-3 << " us." << std::endl;

	std::cout << "Input stats: " << oldh << " rows, " << oldw << " cols." << std::endl;
	std::cout << "Output stats: " << newh << " rows, " << neww << " cols." << std::endl;
}