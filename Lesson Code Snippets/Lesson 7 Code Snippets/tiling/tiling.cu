#include <stdio.h>
#include "gputimer.h"
#include "utils.h"

const int BLOCKSIZE	= 128;
const int NUMBLOCKS = 1000;					// set this to 1 or 2 for debugging
const int N 		= BLOCKSIZE*NUMBLOCKS;

/* 
 * TODO: modify the foo and bar kernels to use tiling: 
 * 		 - copy the input data to shared memory
 *		 - perform the computation there
 *	     - copy the result back to global memory
 *		 - assume thread blocks of 128 threads
 *		 - handle intra-block boundaries correctly
 * You can ignore boundary conditions (we ignore the first 2 and last 2 elements)
 */
__global__ void foo(float out[], float A[], float B[], float C[], float D[], float E[]){

	int globalI = threadIdx.x + blockIdx.x*blockDim.x; 

	int i = threadIdx.x;

	extern __shared__ float shBuffer[];

	int idxA = i;
	int idxB = idxA + blockDim.x;
	int idxC = idxB + blockDim.x;
	int idxD = idxC + blockDim.x;
	int idxE = idxD + blockDim.x;

	shBuffer[idxA] = A[globalI];
	shBuffer[idxB] = B[globalI];
	shBuffer[idxC] = C[globalI];
	shBuffer[idxD] = D[globalI];
	shBuffer[idxE] = E[globalI];
	__syncthreads();
	
	out[globalI] = (shBuffer[idxA] + shBuffer[idxB] + shBuffer[idxC] + shBuffer[idxD] + shBuffer[idxE]) / 5.0f;
}

__global__ void bar(float out[], float in[]) 
{
	int globalI = threadIdx.x + blockIdx.x*blockDim.x; 

	int i = threadIdx.x + 2;

	extern __shared__ float shIn[];

	shIn[i] = in[globalI];
	if (threadIdx.x == 0 && blockIdx.x > 0) {
		shIn[1] = in[globalI - 1];
		shIn[0] = in[globalI - 2];
	} 
	if (threadIdx.x == blockDim.x - 1 && blockIdx.x < gridDim.x) {
		shIn[i + 1] = in[globalI + 1];
		shIn[i + 2] = in[globalI + 2];
	}
	__syncthreads();

	out[globalI] = (shIn[i-2] + shIn[i-1] + shIn[i] + shIn[i+1] + shIn[i+2]) / 5.0f;
}

void cpuFoo(float out[], float A[], float B[], float C[], float D[], float E[])
{
	for (int i=0; i<N; i++)
	{
		out[i] = (A[i] + B[i] + C[i] + D[i] + E[i]) / 5.0f;
	}
}

void cpuBar(float out[], float in[])
{
	// ignore the boundaries
	for (int i=2; i<N-2; i++)
	{
		out[i] = (in[i-2] + in[i-1] + in[i] + in[i+1] + in[i+2]) / 5.0f;
	}
}

int main(int argc, char **argv)
{
	// declare and fill input arrays for foo() and bar()
	float fooA[N], fooB[N], fooC[N], fooD[N], fooE[N], barIn[N];
	for (int i=0; i<N; i++) 
	{
		fooA[i] = i; 
		fooB[i] = i+1;
		fooC[i] = i+2;
		fooD[i] = i+3;
		fooE[i] = i+4;
		barIn[i] = 2*i; 
	}
	// device arrays
	int numBytes = N * sizeof(float);
	float *d_fooA;	 	cudaMalloc(&d_fooA, numBytes);
	float *d_fooB; 		cudaMalloc(&d_fooB, numBytes);
	float *d_fooC;	 	cudaMalloc(&d_fooC, numBytes);
	float *d_fooD; 		cudaMalloc(&d_fooD, numBytes);
	float *d_fooE; 		cudaMalloc(&d_fooE, numBytes);
	float *d_barIn; 	cudaMalloc(&d_barIn, numBytes);
	cudaMemcpy(d_fooA, fooA, numBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_fooB, fooB, numBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_fooC, fooC, numBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_fooD, fooD, numBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_fooE, fooE, numBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_barIn, barIn, numBytes, cudaMemcpyHostToDevice);	

	// output arrays for host and device
	float fooOut[N], barOut[N], *d_fooOut, *d_barOut;
	cudaMalloc(&d_fooOut, numBytes);
	cudaMalloc(&d_barOut, numBytes);

	// declare and compute reference solutions
	float ref_fooOut[N], ref_barOut[N]; 
	cpuFoo(ref_fooOut, fooA, fooB, fooC, fooD, fooE);
	cpuBar(ref_barOut, barIn);

	// launch and time foo and bar
	GpuTimer fooTimer, barTimer;
	const size_t fooShmSize = 5 * BLOCKSIZE * sizeof(float);
	fooTimer.Start();
	foo<<<N/BLOCKSIZE, BLOCKSIZE, fooShmSize>>>(d_fooOut, d_fooA, d_fooB, d_fooC, d_fooD, d_fooE);
	fooTimer.Stop();
	
	const size_t barShmSize = (BLOCKSIZE + 4) * sizeof(float);
	barTimer.Start();
	bar<<<N/BLOCKSIZE, BLOCKSIZE, barShmSize>>>(d_barOut, d_barIn);
	barTimer.Stop();

	cudaMemcpy(fooOut, d_fooOut, numBytes, cudaMemcpyDeviceToHost);
	cudaMemcpy(barOut, d_barOut, numBytes, cudaMemcpyDeviceToHost);
	printf("foo<<<>>>(): %g ms elapsed. Verifying solution...", fooTimer.Elapsed());
	compareArrays(ref_fooOut, fooOut, N);
	printf("bar<<<>>>(): %g ms elapsed. Verifying solution...", barTimer.Elapsed());
	compareArrays(ref_barOut, barOut, N);
}
