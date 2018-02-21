/* Udacity HW5
   Histogramming for Speed

   The goal of this assignment is compute a histogram
   as fast as possible.  We have simplified the problem as much as
   possible to allow you to focus solely on the histogramming algorithm.

   The input values that you need to histogram are already the exact
   bins that need to be updated.  This is unlike in HW3 where you needed
   to compute the range of the data and then do:
   bin = (val - valMin) / valRange to determine the bin.

   Here the bin is just:
   bin = val

   so the serial histogram calculation looks like:
   for (i = 0; i < numElems; ++i)
     histo[val[i]]++;

   That's it!  Your job is to make it run as fast as possible!

   The values are normally distributed - you may take
   advantage of this fact in your implementation.

*/

#include "utils.h"

#include <thrust/adjacent_difference.h>
#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sort.h>

#include <iostream>

__device__ inline unsigned int getPosition() {
    return blockIdx.x * blockDim.x + threadIdx.x;
}

template <typename T>
__global__ void naiveHistoKernel(const T* const vals, unsigned int* const histo, int numVals) {
    unsigned int threadPos = getPosition();

    if (threadPos < numVals) {
        unsigned int bin = vals[threadPos];
        atomicAdd(&histo[bin], 1);
    }
}

template <typename T>
__global__ void shmemHistoKernel(const T* const vals,
                                 unsigned int* const histo,
                                 int numVals,
                                 unsigned int itersPerThread) {
    unsigned int threadPos = getPosition();
    unsigned int threadId = threadIdx.x;

    extern __shared__ unsigned int shHisto[];
    shHisto[threadId] = 0;
    __syncthreads();

    for (int i = 0; i < itersPerThread; i++) {
        threadPos = blockDim.x * (i + itersPerThread * blockIdx.x) + threadIdx.x;
        if (threadPos < numVals) {
            unsigned int bin = vals[threadPos];
            atomicAdd(&shHisto[bin], 1);
        }
    }
    __syncthreads();

    atomicAdd(&histo[threadId], shHisto[threadId]);
}

void thrustHistogram(thrust::device_vector<unsigned int>& vals,
                     unsigned int* histo,
                     int numBins,
                     int numVals) {
    thrust::sort(vals.begin(), vals.end());

    thrust::device_ptr<unsigned int> tHisto = thrust::device_pointer_cast(histo);

    thrust::counting_iterator<unsigned int> searchBegin(0);
    thrust::upper_bound(vals.begin(), vals.end(), searchBegin, searchBegin + numBins, tHisto);

    thrust::adjacent_difference(tHisto, tHisto + numBins, tHisto);
}

void computeHistogram(const unsigned int* const d_vals, // INPUT
                      unsigned int* const d_histo,      // OUTPUT
                      const unsigned int numBins,
                      const unsigned int numElems,
                      const cudaDeviceProp& devProps) {
    std::cout << "numElems = " << numElems << ", numBins = " << numBins << std::endl;

    static constexpr size_t MAX_THREADS_PER_BLOCK = 1024;

    const unsigned int itersPerThread =
        ceil(sqrt(float(numElems) /
                  float(devProps.multiProcessorCount * devProps.maxThreadsPerMultiProcessor)));
    std::cout << "itersPerThread = " << itersPerThread << std::endl;
    const dim3 blocks(1 + (numElems / (MAX_THREADS_PER_BLOCK * itersPerThread)));
    const dim3 threads(MAX_THREADS_PER_BLOCK);

    // Naive histogram - uses global memory and atomic adds
    // naiveHistoKernel<<<blocks, threads>>>(d_vals, d_histo, numElems);

    // Shared memory based implementation - same as before, but computes iteratively:
    // iterations = sqrt(numElems / (<num SMs> * <max num threads per block>))
    const size_t shmSize = threads.x * sizeof(unsigned int);
    shmemHistoKernel<<<blocks, threads, shmSize>>>(d_vals, d_histo, numElems, itersPerThread);

    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());
}
