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

// Prevent memory bank access conflicts
// https://developer.nvidia.com/gpugems/GPUGems3/gpugems3_ch39.html
/*static constexpr size_t NUM_BANKS = 16;
static constexpr size_t LOG_NUM_BANKS = 4;
*/
__device__ inline unsigned int getPosition() {
    return blockIdx.x * blockDim.x + threadIdx.x;
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

template <typename T>
__global__ void predicateKernel(const T* const d_inputVals,
                                const size_t numElems,
                                const unsigned int pass,
                                unsigned int* d_predicates) {
    const unsigned int threadPos = getPosition();

    if (threadPos < numElems) {
        unsigned int mask = 1 << pass;
        d_predicates[threadPos] = d_inputVals[threadPos] & mask == mask ? 1 : 0;
    }
}

// Implementation of Blelloch exclusive scan
// https://developer.nvidia.com/gpugems/GPUGems3/gpugems3_ch39.html
template <typename T>
__global__ void
    exclusiveScanKernel(const T* const d_inputVals, const size_t numElems, T* d_output) {
    extern __shared__ T temp[];

    const unsigned int threadId = threadIdx.x;
    unsigned int offset = 1;

    temp[2 * threadId] = d_inputVals[2 * threadId];
    temp[2 * threadId + 1] = d_inputVals[2 * threadId + 1];

    // Up-sweep (reduce) phase
    for (int d = numElems >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (threadId < d) {
            int ai = offset * (2 * threadId + 1) - 1;
            int bi = offset * (2 * threadId + 2) - 1;
            temp[bi] += temp[ai];
        }
        offset *= 2;
    }

    // Clear last element
    if (threadId == 0) {
        temp[numElems - 1] = 0;
    }

    // Down-sweep phase
    for (int d = 1; d < numElems; d *= 2) {
        offset >>= 1;
        __syncthreads();
        if (threadId < d) {
            int ai = offset * (2 * threadId + 1) - 1;
            int bi = offset * (2 * threadId + 2) - 1;
            T t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
    __syncthreads();

    // Write results to device memory
    d_output[2 * threadId] = temp[2 * threadId];
    d_output[2 * threadId + 1] = temp[2 * threadId + 1];
}

void your_sort(unsigned int* const d_inputVals,
               unsigned int* const d_inputPos,
               unsigned int* const d_outputVals,
               unsigned int* const d_outputPos,
               const size_t numElems) {
    // PUT YOUR SORT HERE
    static constexpr unsigned int MAX_THREADS_PER_BLOCK = 1024;
    unsigned int threads = MAX_THREADS_PER_BLOCK / 2;
    int blocks = numElems / threads + 1;

    // Compute predicate array
    unsigned int* d_predicates;
    const size_t elemBytes = numElems * sizeof(unsigned int);
    checkCudaErrors(cudaMalloc((void**)&d_predicates, elemBytes));
    checkCudaErrors(cudaMemset(d_predicates, 0, elemBytes));

    predicateKernel<<<blocks, threads>>>(d_inputVals, numElems, 0, d_predicates);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());

    // Compute exclusive scan of predicate
    unsigned int* d_scannedPredicate;
    checkCudaErrors(cudaMalloc((void**)&d_scannedPredicate, elemBytes));
    checkCudaErrors(cudaMemset(d_scannedPredicate, 0, elemBytes));

    exclusiveScanKernel<<<blocks, threads>>>(d_predicates, numElems, d_scannedPredicate);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());

    /**uint2* d_inputValsVector;
    const size_t inputVecSize = numElems * sizeof(uint2);
    checkCudaErrors(cudaMalloc((void**)&d_inputValsVector, inputVecSize));

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
    checkCudaErrors(cudaFree(d_scannedPredicate));
    checkCudaErrors(cudaFree(d_predicates));
}
