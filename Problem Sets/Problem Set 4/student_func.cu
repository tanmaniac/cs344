// Udacity HW 4
// Radix Sorting

#include "utils.h"

#include <thrust/host_vector.h>

#include <cfloat>
#include <iostream>

/* Red Eye Removal
   ===============

   For this assignment we are implementing red eye removal.  This is
   accomplished by first creating a score for every pixel that tells us how
   likely it is to be a red eye pixel.  We have already done this for you - you
   are receiving the scores and need to sort them in ascending order so that we
   know which pixels to alter to remove the red eye.

   Note: ascending order == smallest to largest

   Each score is associated with a position, when you sort the scores, you must
   also move the positions accordingly.

   Implementing Parallel Radix Sort with CUDA
   ==========================================

   The basic idea is to construct a histogram on each pass of how many of each
   "digit" there are.   Then we scan this histogram so that we know where to put
   the output of each digit.  For example, the first 1 must come after all the
   0s so we have to know how many 0s there are to be able to start moving 1s
   into the correct position.

   1) Histogram of the number of occurrences of each digit
   2) Exclusive Prefix Sum of Histogram
   3) Determine relative offset of each digit
        For example [0 0 1 1 0 0 1]
                ->  [0 1 0 1 2 3 2]
   4) Combine the results of steps 2 & 3 to determine the final
      output location for each element and move it there

   LSB Radix sort is an out-of-place sort and you will need to ping-pong values
   between the input and output buffers we have provided.  Make sure the final
   sorted results end up in the output buffer!  Hint: You may need to do a copy
   at the end.

 */

template <typename T, size_t count>
struct MakeVector;
template <>
struct MakeVector<unsigned int, 2> {
    typedef uint2 type;
};

__device__ inline unsigned int getPosition() {
    return blockIdx.x * blockDim.x * threadIdx.x;
}

template <typename T>
__device__ inline T cuda_min(T a, T b) {
    return a < b ? a : b;
}

template <typename T>
__device__ inline T cuda_max(T a, T b) {
    return a > b ? a : b;
}

template <typename T>
__global__ void copyToVec2(T* const d_inputVals,
                           const size_t numElems,
                           typename MakeVector<T, 2>::type* d_inputValsVector) {
    static_assert(std::is_arithmetic<T>::value, "Only arithmetic types are supported");

    unsigned int threadPos = getPosition();
    if (threadPos < numElems) {
        d_inputValsVector[threadPos].x = d_inputVals[threadPos];
        d_inputValsVector[threadPos].y = d_inputVals[threadPos];
    }
}

template <typename T>
__global__ void minMaxKernel(T* d_inputValsVector,
                             const size_t numElems) {
    //static_assert(std::is_arithmetic<T>::value, "Only arithmetic types are supported");
    const int threadPos = getPosition();
    const int threadId = threadIdx.x;

    // Shared memory is an array of 2-vector types so we can interleave min and max values
    //typedef typename MakeVector<T, 2>::type vec2_type;
    extern __shared__ T shmMinMaxVals[];

    // Copy values into shared memory. FLT_MAX is default value for min vals (everything is
    // lower) and -FLT_MAX is default for max vals (everything is higher)
    shmMinMaxVals[threadId].x = threadPos < numElems ? d_inputValsVector[threadPos].x : FLT_MAX;
    shmMinMaxVals[threadId].y = threadPos < numElems ? d_inputValsVector[threadPos].y : -FLT_MAX;
    __syncthreads();

    // Reduce this block
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadId < s) {
            shmMinMaxVals[threadId].x =
                cuda_min(shmMinMaxVals[threadId].x, shmMinMaxVals[threadId + s].x);
            shmMinMaxVals[threadId].y =
                cuda_max(shmMinMaxVals[threadId].y, shmMinMaxVals[threadId + s].y);
        }
        __syncthreads();
    }

    // Only write to output array with thread 0
    if (threadId == 0) {
        d_inputValsVector[blockIdx.x].x = shmMinMaxVals[0].x;
        d_inputValsVector[blockIdx.x].y = shmMinMaxVals[0].y;
    }
}

// Round up to the nearest power of two
// https://graphics.stanford.edu/~seander/bithacks.html#RoundUpPowerOf2
uint32_t roundUpToPow2(uint32_t val) {
    val--;
    val |= val >> 1;
    val |= val >> 2;
    val |= val >> 4;
    val |= val >> 8;
    val |= val >> 16;
    val++;
    return val;
}

void your_sort(unsigned int* const d_inputVals,
               unsigned int* const d_inputPos,
               unsigned int* const d_outputVals,
               unsigned int* const d_outputPos,
               const size_t numElems) {
    // TODO
    // PUT YOUR SORT HERE
    uint2* d_inputValsVector;
    const size_t inputVecSize = numElems * sizeof(uint2);
    checkCudaErrors(cudaMalloc((void**)&d_inputValsVector, inputVecSize));

    static constexpr unsigned int MAX_THREADS_PER_BLOCK = 1024;
    unsigned int threads = MAX_THREADS_PER_BLOCK;
    int blocks = numElems / threads + 1;

    copyToVec2<<<blocks, threads>>>(d_inputVals, numElems, d_inputValsVector);

    // Run min/max kernel on the dataset and reduce all the threads
    size_t shmSize = threads * sizeof(uint2);
    minMaxKernel<<<blocks, threads, shmSize>>>(d_inputValsVector, numElems);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());

    // Want a power of 2 for the final reduce step
    threads = roundUpToPow2(blocks);
    // Use the previous number of blocks as the new number of elements
    size_t prevNumBlocks = blocks;
    blocks = 1;
    shmSize = threads * sizeof(uint2);
    minMaxKernel<<<blocks, threads, shmSize>>>(d_inputValsVector, prevNumBlocks);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());

    unsigned int min = d_inputValsVector[0].x;
    unsigned int max = d_inputValsVector[0].y;

    std::cout << "Min = " << min << " Max = " << max << std::endl;
    /***** Clean up *****/
    checkCudaErrors(cudaFree(d_inputValsVector));
}
