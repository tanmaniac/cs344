// Udacity HW 4
// Radix Sorting

#include "utils.h"

#include <cuda_runtime.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>

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

__device__ inline unsigned int getPosition() {
    return blockIdx.x * blockDim.x + threadIdx.x;
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

// Count the number of occurrences of each number with some number of bits in the pass'th bit of
// each value in the input array.
template <typename T>
__global__ void histoKernel(const T* const input,
                            const size_t inputSize,
                            const unsigned int pass,
                            unsigned int* histo) {
    const unsigned int threadPos = getPosition();
    const unsigned int predicate = 1 << pass;

    if (threadPos < inputSize) {
        unsigned int bin = ((input[threadPos] & predicate) == predicate) ? 1 : 0;
        atomicAdd(&histo[bin], 1);
    }
}

template <typename T>
__global__ void predicateKernel(const T* const d_input,
                                const size_t inputSize,
                                unsigned int pass,
                                unsigned int* d_output) {
    const unsigned int threadPos = getPosition();
    const unsigned int predicate = 1 << pass;

    if (threadPos < inputSize) {
        unsigned int masked = ((d_input[threadPos] & predicate) == predicate) ? 1 : 0;
        d_output[threadPos] = masked;
    }
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

__global__ void uniformNotKernel(const unsigned int* const d_dataIn,
                                 const size_t dataSize,
                                 unsigned int* d_dataOut) {
    const unsigned int threadPos = getPosition();
    if (threadPos < dataSize) {
        d_dataOut[threadPos] = !(d_dataIn[threadPos]);
    }
}

template <typename T>
__global__ void scatterKernel(const T* const d_inputVals,
                              const unsigned int* const d_inputPos,
                              const unsigned int* const d_predicate,
                              const unsigned int* const d_scannedPredicate,
                              const size_t dataSize,
                              const size_t offset,
                              T* d_outputVals,
                              unsigned int* d_outputPos) {
    const unsigned int threadPos = getPosition();

    if (threadPos < dataSize) {
        if (d_predicate[threadPos]) {
            unsigned int outputIndex = d_scannedPredicate[threadPos] + offset;
            d_outputVals[outputIndex] = d_inputVals[threadPos];
            d_outputPos[outputIndex] = d_inputPos[threadPos];
        }
    }
}

template <typename T>
__global__ void swapBuffersKernel(const T* const d_inputVals,
                                  const unsigned int* const d_inputPos,
                                  const size_t dataSize,
                                  T* d_outputVals,
                                  unsigned int* d_outputPos) {
    const unsigned int threadPos = getPosition();

    if (threadPos < dataSize) {
        d_outputPos[threadPos] = d_inputPos[threadPos];
        d_outputVals[threadPos] = d_inputVals[threadPos];
    }
}

void runExclusiveScan(unsigned int* d_predicate,
                      unsigned int blocksPass1,
                      unsigned int threadsPass1,
                      unsigned int blocksPass2,
                      unsigned int threadsPass2,
                      size_t dataSize,
                      size_t shmSize,
                      unsigned int* d_intermediate,
                      unsigned int* d_dataOut,
                      unsigned int* d_sums,
                      unsigned int* d_incr) {
    // Determine if we even need to do two passes or not. If one more more threads are necessary to
    // compute the sums scan, then we definitely need to do two passes
    if (threadsPass2 > 0) {
        // Scan blocks at the top level, storing the output in d_intermediate and the block-level
        // sums in d_sums
        bool isBlockLevelScan = true;
        exclusiveBlellochKernel<<<blocksPass1, threadsPass1, shmSize>>>(d_predicate,
                                                                        dataSize / blocksPass1,
                                                                        dataSize,
                                                                        isBlockLevelScan,
                                                                        d_intermediate,
                                                                        d_sums);
        // Scan the array of block sums to determine how much to add to each block
        const size_t sumsShmSize = blocksPass1 * sizeof(unsigned int);
        isBlockLevelScan = false;
        exclusiveBlellochKernel<<<blocksPass2, threadsPass2, sumsShmSize>>>(
            d_sums, blocksPass1, blocksPass1, isBlockLevelScan, d_incr, d_incr);
        // Add d_incr to the intermediate sums to get the actual output
        uniformAddKernel<<<blocksPass1, (dataSize / blocksPass1)>>>(
            d_intermediate, dataSize, d_incr, d_dataOut);
    } else {
        // No need to do the second pass so don't use an intermediate array
        exclusiveBlellochKernel<<<blocksPass1, threadsPass1, shmSize>>>(
            d_predicate, dataSize / blocksPass1, dataSize, false, d_dataOut, d_sums);
    }
}

void copyAndPrint(unsigned int* d_predicate,
                  unsigned int* d_notPredicate,
                  unsigned int* d_dataOut,
                  unsigned int* d_inputVals) {
    std::array<unsigned int, 100> predicate, notPredicate, dataOut, inputVals;
    const size_t byteSize = 100 * sizeof(unsigned int);

    checkCudaErrors(cudaMemcpy(predicate.data(), d_predicate, byteSize, cudaMemcpyDeviceToHost));
    checkCudaErrors(
        cudaMemcpy(notPredicate.data(), d_notPredicate, byteSize, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(dataOut.data(), d_dataOut, byteSize, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(inputVals.data(), d_inputVals, byteSize, cudaMemcpyDeviceToHost));

    std::cout << "predicate: [ ";
    for (const auto& val : predicate) {
        std::cout << val << " ";
    }
    std::cout << "]" << std::endl;

    std::cout << "notPredicate: [ ";
    for (const auto& val : notPredicate) {
        std::cout << val << " ";
    }
    std::cout << "]" << std::endl;

    std::cout << "dataOut: [ ";
    for (const auto& val : dataOut) {
        std::cout << val << " ";
    }
    std::cout << "]" << std::endl;

    std::cout << "inputVals: [ ";
    for (const auto& val : inputVals) {
        std::cout << val << " ";
    }
    std::cout << "]" << std::endl;
}

void your_sort(unsigned int* const d_inputVals,
               unsigned int* const d_inputPos,
               unsigned int* const d_outputVals,
               unsigned int* const d_outputPos,
               const size_t numElems) {
    // PUT YOUR SORT HERE
    static constexpr size_t MAX_NUM_THREADS = 256;
    const size_t dataSize = isPow2(numElems) ? numElems : roundUpToPow2(numElems);
    const size_t dataBytes = dataSize * sizeof(unsigned int);
    const size_t bins = 2;

    // Blocks and threads for Blelloch kernels
    // Determine how many blocks and threads to run for each pass
    // First pass = scanning each block separately
    // Second pass = scanning the sum of each block as computed in the first pass
    unsigned int blocksPass1 =
        max(1, (unsigned int)ceil(float(dataSize) / (2.f * MAX_NUM_THREADS)));
    unsigned int threadsPass1 = min(dataSize, MAX_NUM_THREADS);
    unsigned int blocksPass2 = 1;
    unsigned int threadsPass2 = blocksPass1 / 2;

    // Blocks and threads for everything else
    // Histo, predicate, etc operate on all of the elements of the input and not just half of them

    // Declare GPU memory pointers
    unsigned int *d_predicate, *d_notPredicate, *d_dataOut, *d_intermediate, *d_sums, *d_incr,
        *d_histo;

    // Allocate GPU memory
    checkCudaErrors(cudaMalloc((void**)&d_dataOut, dataBytes));
    checkCudaErrors(cudaMalloc((void**)&d_intermediate, dataBytes));
    checkCudaErrors(cudaMalloc((void**)&d_predicate, dataBytes));
    checkCudaErrors(cudaMalloc((void**)&d_notPredicate, dataBytes));
    checkCudaErrors(cudaMalloc((void**)&d_histo, bins * sizeof(unsigned int)));

    checkCudaErrors(cudaMemset(d_dataOut, 0, dataBytes));
    checkCudaErrors(cudaMemset(d_intermediate, 0, dataBytes));
    checkCudaErrors(cudaMemset(d_predicate, 0, dataBytes));
    checkCudaErrors(cudaMemset(d_notPredicate, 0, dataBytes));

    // Allocate memory for block sums and scanned block sums
    const size_t blockBytes = blocksPass1 * sizeof(int);
    checkCudaErrors(cudaMalloc((void**)&d_sums, blockBytes));
    checkCudaErrors(cudaMalloc((void**)&d_incr, blockBytes));

    // Determine space required for shared memory
    // unsigned int extraSpace = dataSize / blocks / NUM_BANKS;
    // unsigned int shmSize = (dataSize / blocks + extraSpace) * sizeof(int);
    const size_t shmSize = threadsPass1 * 2 * sizeof(int);

    // Needs to be done 32 times (!!!)
    for (unsigned int pass = 0; pass < 8 * sizeof(unsigned int); pass++) {
        // Reset values in histo array
        checkCudaErrors(cudaMemset(d_histo, 0, bins * sizeof(unsigned int)));
        // Compute offsets
        histoKernel<<<blocksPass1, 2 * threadsPass1>>>(d_inputVals, numElems, pass, d_histo);
        cudaDeviceSynchronize();
        checkCudaErrors(cudaGetLastError());
        std::array<unsigned int, bins> histo;
        checkCudaErrors(
            cudaMemcpy(histo.data(), d_histo, bins * sizeof(unsigned int), cudaMemcpyDeviceToHost));

        predicateKernel<<<blocksPass1, 2 * threadsPass1>>>(
            d_inputVals, numElems, pass, d_predicate);
        cudaDeviceSynchronize();
        checkCudaErrors(cudaGetLastError());

        uniformNotKernel<<<blocksPass1, 2 * threadsPass1>>>(d_predicate, dataSize, d_notPredicate);
        cudaDeviceSynchronize();
        checkCudaErrors(cudaGetLastError());

        runExclusiveScan(d_predicate,
                         blocksPass1,
                         threadsPass1,
                         blocksPass2,
                         threadsPass2,
                         dataSize,
                         shmSize,
                         d_intermediate,
                         d_dataOut,
                         d_sums,
                         d_incr);
        cudaDeviceSynchronize();
        checkCudaErrors(cudaGetLastError());

        scatterKernel<<<blocksPass1, 2 * threadsPass1>>>(d_inputVals,
                                                         d_inputPos,
                                                         d_predicate,
                                                         d_dataOut,
                                                         numElems,
                                                         histo[0],
                                                         d_outputVals,
                                                         d_outputPos);
        cudaDeviceSynchronize();
        checkCudaErrors(cudaGetLastError());
        runExclusiveScan(d_notPredicate,
                         blocksPass1,
                         threadsPass1,
                         blocksPass2,
                         threadsPass2,
                         dataSize,
                         shmSize,
                         d_intermediate,
                         d_dataOut,
                         d_sums,
                         d_incr);
        cudaDeviceSynchronize();
        checkCudaErrors(cudaGetLastError());

        scatterKernel<<<blocksPass1, 2 * threadsPass1>>>(d_inputVals,
                                                         d_inputPos,
                                                         d_notPredicate,
                                                         d_dataOut,
                                                         numElems,
                                                         0,
                                                         d_outputVals,
                                                         d_outputPos);
        cudaDeviceSynchronize();
        checkCudaErrors(cudaGetLastError());
        swapBuffersKernel<<<blocksPass1, 2 * threadsPass1>>>(
            d_outputVals, d_outputPos, numElems, d_inputVals, d_inputPos);
        cudaDeviceSynchronize();
        checkCudaErrors(cudaGetLastError());

        // copyAndPrint(d_predicate, d_notPredicate, d_dataOut, d_inputVals);
    }

    // Free memory
    checkCudaErrors(cudaFree(d_dataOut));
    checkCudaErrors(cudaFree(d_intermediate));
    checkCudaErrors(cudaFree(d_sums));
    checkCudaErrors(cudaFree(d_incr));
    checkCudaErrors(cudaFree(d_predicate));
    checkCudaErrors(cudaFree(d_histo));
}

// Thrust's radix sort - performance benchmark (pretty much as fast as you can possibly go).
// Measured performance on Quadro M1000M: ~1.208 ms
void thrustSort(unsigned int* const inputVals,
                unsigned int* const inputPos,
                unsigned int* const outputVals,
                unsigned int* const outputPos,
                const size_t numElems) {
    // make device pointers to wrap our raw pointers
    thrust::device_ptr<unsigned int> inputValsPtr = thrust::device_pointer_cast(inputVals);
    thrust::device_ptr<unsigned int> inputPosPtr = thrust::device_pointer_cast(inputPos);
    // Do a stable sort where the input values are keys and the input positions are values
    thrust::stable_sort_by_key(inputValsPtr, inputValsPtr + numElems, inputPosPtr);
    // Copy everything into the output arrays
    thrust::copy(inputValsPtr, inputValsPtr + numElems, outputVals);
    thrust::copy(inputPosPtr, inputPosPtr + numElems, outputPos);
}
