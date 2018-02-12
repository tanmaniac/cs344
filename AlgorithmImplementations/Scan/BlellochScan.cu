/**
 *  Implementation of Blelloch parallel prefix scan.
 *  https://www.cs.cmu.edu/~blelloch/papers/Ble93.pdf
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <limits>
#include <type_traits>

static constexpr unsigned int NUM_BANKS = 16;
static constexpr unsigned int LOG_NUM_BANKS = 4;

__device__ inline int getPosition() {
    return blockIdx.x * blockDim.x + threadIdx.x;
}

template <typename T>
__device__ inline T conflictFreeOffset(T index) {
    static_assert(std::numeric_limits<T>::is_integer, "Only integer types are supported");
    return index >> LOG_NUM_BANKS + index >> (2 * LOG_NUM_BANKS);
}

// Implementation of Blelloch exclusive scan
// https://developer.nvidia.com/gpugems/GPUGems3/gpugems3_ch39.html
template <typename T>
__global__ void exclusiveBlellochKernel(const T* const d_inputVals,
                                        const size_t numElems,
                                        const bool isBlockLevelScan,
                                        T* d_output,
                                        T* d_sums) {
    static_assert(std::is_arithmetic<T>::value, "Only arithmetic types are supported");
    extern __shared__ T temp[];

    const unsigned int threadPos = getPosition();
    const unsigned int threadId = threadIdx.x;
    //printf("Thread = %d, threadPos = %d\n", threadId, threadPos);
    unsigned int offset = 1;

    /*int ai = threadId;
    int bi = threadId + (numElems / 2);

    // Avoid memory bank conflicts
    int bankOffsetA = conflictFreeOffset(ai);
    int bankOffsetB = conflictFreeOffset(bi);

    // Copy into shared memory
    temp[ai + bankOffsetA] = d_inputVals[ai + blockOffset];
    temp[bi + bankOffsetB] = d_inputVals[bi + blockOffset];
    */
    // Block A in code sample
    temp[2 * threadId] = d_inputVals[2 * threadPos];
    temp[2 * threadId + 1] = d_inputVals[2 * threadPos + 1];
    printf("Thread %d, temp[%d] = %d, temp[%d] = %d\n",
           threadId,
           2 * threadId,
           d_inputVals[2 * threadPos],
           2 * threadId + 1,
           d_inputVals[2 * threadPos + 1]);

    // Up-sweep (reduce) phase
    for (int d = numElems >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (threadId < d) {
            /*ai = offset * (2 * threadId + 1) - 1;
            bi = offset * (2 * threadId + 2) - 1;
            ai += conflictFreeOffset(ai);
            bi += conflictFreeOffset(bi);
            */
            // Block B in code sample
            int ai = offset * (2 * threadId + 1) - 1;
            int bi = offset * (2 * threadId + 2) - 1;
            if (!isBlockLevelScan) {
                printf("Thread %d: ai = %d, bi = %d\n", threadId, ai, bi);
            }
            temp[bi] += temp[ai];
        }
        offset *= 2;
    }

    if (isBlockLevelScan) {
        d_sums[blockIdx.x] = temp[numElems-1];
    }

    // Clear last element
    if (threadId == 0) {
        temp[numElems - 1] = 0;
        // temp[numElems - 1 + conflictFreeOffset(numElems - 1)] = 0;
    }

    // Down-sweep phase
    for (int d = 1; d < numElems; d *= 2) {
        offset >>= 1;
        __syncthreads();
        if (threadId < d) {
            /*ai = offset * (2 * threadId + 1) - 1;
            bi = offset * (2 * threadId + 2) - 1;
            ai += conflictFreeOffset(ai);
            bi += conflictFreeOffset(bi);
            */
            // Block D in code sample
            int ai = offset * (2 * threadId + 1) - 1;
            int bi = offset * (2 * threadId + 2) - 1;
            T t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
    __syncthreads();

    // Write results to device memory
    d_output[2 * threadPos] = temp[2 * threadId];
    d_output[2 * threadPos + 1] = temp[2 * threadId + 1];
    // d_output[ai + blockOffset] = temp[ai + bankOffsetA];
    // d_output[bi + blockOffset] = temp[bi + bankOffsetB];
}

template <typename T>
__global__ void uniformAddKernel(T* d_dataIn, const size_t dataSize, T* d_incr, T* d_dataOut) {
    const unsigned int threadPos = getPosition();
    const unsigned int blockId = blockIdx.x;

    if (threadPos < dataSize) {
        d_dataOut[threadPos] = d_dataIn[threadPos] + d_incr[blockId];
    }
}

// Check if input is a power of two
bool isPow2(int n) {
    return n && (!(n & (n - 1)));
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

void launchBlellochKernel(int* h_dataOut,
                          const int* h_dataIn,
                          const size_t dataSize,
                          float* execTime = nullptr) {
    const size_t dataBytes = dataSize * sizeof(int);

    // Declare GPU memory pointers
    int *d_dataOut, *d_intermediate, *d_dataIn, *d_sums, *d_incr;

    // Set up GPU timers
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Allocate GPU memory
    cudaMalloc((void**)&d_dataOut, dataBytes);
    cudaMalloc((void**)&d_dataIn, dataBytes);
    cudaMalloc((void**)&d_intermediate, dataBytes);

    cudaMemcpy(d_dataIn, h_dataIn, dataBytes, cudaMemcpyHostToDevice);

    unsigned int blocks = 4;
    unsigned int threads = dataSize / blocks / 2;

    // Allocate memory for block sums and scanned block sums
    const size_t blockBytes = blocks * sizeof(int);
    cudaMalloc((void**)&d_sums, blockBytes);
    cudaMalloc((void**)&d_incr, blockBytes);

    // Determine space required for shared memory
    // unsigned int extraSpace = dataSize / blocks / NUM_BANKS;
    // unsigned int shmSize = (dataSize / blocks + extraSpace) * sizeof(int);
    const size_t shmSize = (dataSize / blocks) * sizeof(int);

    // Execute kernel and record runtime
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    // Scan blocks at the top level, storing the output in d_intermediate and the block-level sums in d_sums
    bool isBlockLevelScan = true;
    exclusiveBlellochKernel<<<blocks, threads, shmSize>>>(d_dataIn, dataSize / blocks, isBlockLevelScan, d_intermediate, d_sums);
    // Scan the array of block sums to determine how much to add to each block
    // New number of threads = old number of blocks
    unsigned int sumsThreads = blocks / 2;
    unsigned int sumsBlocks = 1;
    const size_t sumsShmSize = blocks * sizeof(int);
    exclusiveBlellochKernel<<<sumsBlocks, sumsThreads, sumsShmSize>>>(d_sums, blocks, false, d_incr, d_incr);
    // Add d_incr to the intermediate sums to get the actual output
    uniformAddKernel<<<blocks, (dataSize / blocks)>>>(d_intermediate, dataSize, d_incr, d_dataOut);
    cudaEventRecord(stop);
    cudaDeviceSynchronize();

    // Copy back from GPU to CPU
    cudaMemcpy(h_dataOut, d_dataOut, dataBytes, cudaMemcpyDeviceToHost);

    cudaEventSynchronize(stop);
    float timeInMs = 0;
    cudaEventElapsedTime(&timeInMs, start, stop);
    *execTime = timeInMs;

    // Free memory
    cudaFree(d_dataIn);
    cudaFree(d_dataOut);
    cudaFree(d_intermediate);
    cudaFree(d_sums);
    cudaFree(d_incr);
}
