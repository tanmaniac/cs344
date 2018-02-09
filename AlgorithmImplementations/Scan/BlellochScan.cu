/**
 *  Implementation of Blelloch parallel prefix scan.
 *  https://www.cs.cmu.edu/~blelloch/papers/Ble93.pdf
 */

 #include <stdio.h>
 #include <stdlib.h>
 #include <cuda_runtime.h>

 // Implementation of Blelloch exclusive scan
// https://developer.nvidia.com/gpugems/GPUGems3/gpugems3_ch39.html
template <typename T>
__global__ void
    exclusiveBlellochKernel(const T* const d_inputVals, const size_t numElems, T* d_output) {
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

void launchBlellochKernel(int *h_dataOut, const int *h_dataIn, const size_t dataSize, float *execTime = nullptr) {
    const size_t dataBytes = dataSize * sizeof(int);

    // Declare GPU memory pointers
    int *d_dataOut, *d_dataIn;

    // Set up GPU timers
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Allocate GPU memory
    cudaMalloc((void **) &d_dataOut, dataBytes);
    cudaMalloc((void **) &d_dataIn, dataBytes);

    cudaMemcpy(d_dataIn, h_dataIn, dataBytes, cudaMemcpyHostToDevice);

    // Execute kernel and record runtime
    cudaEventRecord(start);
    exclusiveBlellochKernel<<<1, dataSize/2, 2 * dataBytes>>>(d_dataIn, dataSize, d_dataOut);
    cudaEventRecord(stop);

    // Copy back from GPU to CPU
    cudaMemcpy(h_dataOut, d_dataOut, dataBytes, cudaMemcpyDeviceToHost);

    cudaEventSynchronize(stop);
    float timeInMs = 0;
    cudaEventElapsedTime(&timeInMs, start, stop);
    *execTime = timeInMs;

    // Free memory
    cudaFree(d_dataIn);
    cudaFree(d_dataOut);
}
