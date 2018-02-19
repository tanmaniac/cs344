/**
 *  Implementation of Blelloch parallel prefix scan.
 *  https://www.cs.cmu.edu/~blelloch/papers/Ble93.pdf
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <cstdint>
#include <iostream>
#include <limits>
#include <type_traits>

__device__ inline int getPosition() {
    return blockIdx.x * blockDim.x + threadIdx.x;
}

template <typename T>
__device__ void
    copyGlobalToSharedMem(const T* const d_inputVals, const size_t fullDataSize, T* s_vals) {
    const unsigned int threadPos = getPosition();
    const unsigned int threadId = threadIdx.x;

    // Block A in code sample
    s_vals[2 * threadId] = (2 * threadPos < fullDataSize) ? d_inputVals[2 * threadPos] : 0;
    s_vals[2 * threadId + 1] =
        (2 * threadPos + 1 < fullDataSize) ? d_inputVals[2 * threadPos + 1] : 0;
}

// Does upsweep step in place
template <typename T>
__device__ unsigned int upsweep(T* s_vals, const size_t numElems) {
    const unsigned int threadId = threadIdx.x;

    unsigned int offset = 1;

    for (int d = numElems >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (threadId < d) {
            int ai = offset * (2 * threadId + 1) - 1;
            int bi = offset * (2 * threadId + 2) - 1;
            s_vals[bi] += s_vals[ai];
        }
        offset *= 2;
    }
    return offset;
}

// Copy last element into the sums array if necessary, and then clears the last element of the input
template <typename T>
__device__ void
    clearLastElement(T* s_vals, const size_t numElems, const bool isBlockLevelScan, T* d_sums) {
    if (isBlockLevelScan) {
        d_sums[blockIdx.x] = s_vals[numElems - 1];
    }
    s_vals[numElems - 1] = 0;
}

// Downsweep step in place
template <typename T>
__device__ void downsweep(T* s_vals, const size_t numElems, unsigned int offset) {
    const unsigned int threadId = threadIdx.x;

    for (int d = 1; d < numElems; d *= 2) {
        offset >>= 1;
        __syncthreads();
        if (threadId < d) {
            int ai = offset * (2 * threadId + 1) - 1;
            int bi = offset * (2 * threadId + 2) - 1;
            T t = s_vals[ai];
            s_vals[ai] = s_vals[bi];
            s_vals[bi] += t;
        }
    }
}

template <typename T>
__device__ void
    copySharedMemToGlobal(const T* const s_vals, const size_t fullDataSize, T* d_output) {
    const unsigned int threadPos = getPosition();
    const unsigned int threadId = threadIdx.x;

    unsigned int outputAddr = 2 * threadPos;
    if (outputAddr < fullDataSize) {
        d_output[outputAddr] = s_vals[2 * threadId];
    }
    if (outputAddr + 1 < fullDataSize) {
        d_output[outputAddr + 1] = s_vals[2 * threadId + 1];
    }
}

// Implementation of Blelloch exclusive scan
// https://developer.nvidia.com/gpugems/GPUGems3/gpugems3_ch39.html
template <typename T>
__global__ void exclusiveBlellochKernel(const T* const d_inputVals,
                                        const size_t numElems,
                                        const size_t fullDataSize,
                                        const bool isBlockLevelScan,
                                        T* d_output,
                                        T* d_sums) {
    static_assert(std::is_arithmetic<T>::value, "Only arithmetic types are supported");
    extern __shared__ T temp[];

    const unsigned int threadPos = getPosition();
    const unsigned int threadId = threadIdx.x;
    // printf("Thread = %d, threadPos = %d\n", threadId, threadPos);

    copyGlobalToSharedMem(d_inputVals, fullDataSize, temp);
    __syncthreads();

    // Up-sweep (reduce) phase
    unsigned int offset = upsweep(temp, numElems);
    __syncthreads();

    // Clear last element
    if (threadId == 0) {
        clearLastElement(temp, numElems, isBlockLevelScan, d_sums);
    }
    __syncthreads();

    // Down-sweep phase
    downsweep(temp, numElems, offset);
    __syncthreads();

    // Write results to device memory
    copySharedMemToGlobal(temp, fullDataSize, d_output);
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
                          const size_t realDataSize,
                          float* execTime = nullptr) {
    static constexpr size_t MAX_NUM_THREADS = 256;

    const size_t dataSize = isPow2(realDataSize) ? realDataSize : roundUpToPow2(realDataSize);
    // Determine how many blocks and threads to run for each pass
    // First pass = scanning each block separately
    // Second pass = scanning the sum of each block as computed in the first pass
    unsigned int blocksP1 = max(1, (unsigned int)ceil(float(dataSize) / (2.f * MAX_NUM_THREADS)));
    unsigned int threadsP1 = min(dataSize, MAX_NUM_THREADS);
    unsigned int blocksP2 = 1;
    unsigned int threadsP2 = blocksP1 / 2;
    //std::cout << "BlocksP1 = " << blocksP1 << " threadsP1 = " << threadsP1 << " blocksP2 = " << blocksP2 << " threadsP2 = " << threadsP2 << std::endl;

    const size_t dataBytes = dataSize * sizeof(int);
    //const size_t dataBytes = threadsP1 * sizeof(int);

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
    cudaMemset(d_dataOut, 0, dataBytes);
    cudaMemset(d_dataIn, 0, dataBytes);
    cudaMemset(d_intermediate, 0, dataBytes);

    cudaMemcpy(d_dataIn, h_dataIn, realDataSize * sizeof(int), cudaMemcpyHostToDevice);

    // Allocate memory for block sums and scanned block sums
    const size_t blockBytes = blocksP1 * sizeof(int);
    cudaMalloc((void**)&d_sums, blockBytes);
    cudaMalloc((void**)&d_incr, blockBytes);

    // Determine space required for shared memory
    // unsigned int extraSpace = dataSize / blocks / NUM_BANKS;
    // unsigned int shmSize = (dataSize / blocks + extraSpace) * sizeof(int);
    const size_t shmSize = threadsP1 * 2 * sizeof(int);

    // Execute kernel and record runtime
    cudaDeviceSynchronize();
    cudaEventRecord(start);

    // Determine if we even need to do two passes or not. If one more more threads are necessary to
    // compute the sums scan, then we definitely need to do two passes
    if (threadsP2 > 0) {
        // Scan blocks at the top level, storing the output in d_intermediate and the block-level
        // sums in d_sums
        bool isBlockLevelScan = true;
        exclusiveBlellochKernel<<<blocksP1, threadsP1, shmSize>>>(
            d_dataIn, dataSize / blocksP1, dataSize, isBlockLevelScan, d_intermediate, d_sums);
        // Scan the array of block sums to determine how much to add to each block
        const size_t sumsShmSize = blocksP1 * sizeof(int);
        isBlockLevelScan = false;
        exclusiveBlellochKernel<<<blocksP2, threadsP2, sumsShmSize>>>(
            d_sums, blocksP1, blocksP1, isBlockLevelScan, d_incr, d_incr);
        // Add d_incr to the intermediate sums to get the actual output
        uniformAddKernel<<<blocksP1, (dataSize / blocksP1)>>>(
            d_intermediate, dataSize, d_incr, d_dataOut);
    } else {
        // No need to do the second pass so don't use an intermediate array
        exclusiveBlellochKernel<<<blocksP1, threadsP1, shmSize>>>(
            d_dataIn, dataSize / blocksP1, dataSize, false, d_dataOut, d_sums);
    }
    cudaEventRecord(stop);
    cudaDeviceSynchronize();

    // Copy back from GPU to CPU
    cudaMemcpy(h_dataOut, d_dataOut, realDataSize * sizeof(int), cudaMemcpyDeviceToHost);

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
