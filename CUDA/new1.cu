#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define N 4
#define thread_num 4
#define block_num 2

__global__ void prescan(float *g_odata, float *g_idata, int n);

void scanCPU(float *f_out, float *f_in, int i_n);

double myDiffTime(struct timeval &start, struct timeval &end)
{
	double d_start, d_end;
	d_start = (double)(start.tv_sec + start.tv_usec/1000000.0);
	d_end = (double)(end.tv_sec + end.tv_usec/1000000.0);
	return (d_end - d_start);
}


__global__ void prescan(float *g_odata, float *g_idata, int n)
{
	extern  __shared__  float temp[];
	// allocated on invocation
	int thid = threadIdx.x;
	int bid = blockIdx.x;
	int fake_n = block_num * thread_num;


	int offset = 0;
	if(bid * thread_num + thid<n)
		{
	 		temp[bid * thread_num + thid]  = g_idata[bid * thread_num + thid];
		}
	else
	{
	 temp[bid * thread_num + thid]  = 0;
		} // Make the "empty" spots zeros, so it won't affect the final result.

	for (int d = thread_num>>1; d > 0; d >>= 1)
    // build sum in place up the tree
	{
    	__syncthreads();
    if (thid < d)
    {
        int ai = bid * thread_num + offset*(2*thid+1)-1;
        int bi = bid * thread_num + offset*(2*thid+2)-1;
        temp[bi] += temp[ai];
    }
    offset *= 2;
}

if (thid == 0)
{
    temp[thread_num - 1] = 0;
}

// clear the last element
for (int d = 1; d < thread_num; d *= 2)
    // traverse down tree & build scan
{
    offset >>= 1;
    __syncthreads();

  if (thid < d)
    {
        int ai = bid * thread_num + offset*(2*thid+1)-1;
        int bi = bid * thread_num + offset*(2*thid+2)-1;
        float t = temp[bid * thread_num + ai];
        temp[ai]  = temp[ bi];
        temp[bi] += t;
    }
}
__syncthreads();

g_odata[bid * thread_num + thid] = temp[bid * thread_num + thid];
}

void scanCPU(float *f_out, float *f_in, int i_n)
	{
		f_out[0] = 0;
		for (int i = 1; i <= i_n; i++)
    		f_out[i] = f_out[i-1] + f_in[i-1];

		}



int main()
{
	float a[N], c[N], g[N];
	timeval start, end;
	float *dev_a, *dev_g;
	int size = N * sizeof(float);
	double d_gpuTime, d_cpuTime;

for (int i = 0; i < N; i++)
{
    	a[i] = i;
    	printf("a[%i] = %f\n", i, a[i]);
}

// initialize a and b matrices here
cudaMalloc((void **) &dev_a, size);
cudaMalloc((void **) &dev_g, size);

gettimeofday(&start, NULL);

cudaMemcpy(dev_a, a, size, cudaMemcpyHostToDevice);

prescan<<<block_num,thread_num,2*N*sizeof(float)>>>(dev_g, dev_a, N);
cudaDeviceSynchronize();

cudaMemcpy(g, dev_g, size, cudaMemcpyDeviceToHost);

gettimeofday(&end, NULL);
d_gpuTime = myDiffTime(start, end);

gettimeofday(&start, NULL);
scanCPU(c, a, N);

gettimeofday(&end, NULL);
d_cpuTime = myDiffTime(start, end);

cudaFree(dev_a); cudaFree(dev_g);

for (int i = 1; i <= N; i++)
{
    printf("c[%i] = %0.3f, g[%i] = %0.3f\n", i, c[i], i, g[i]);
}

printf("GPU Time for scan size %i: %f\n", N, d_gpuTime);
printf("CPU Time for scan size %i: %f\n", N, d_cpuTime);
}


