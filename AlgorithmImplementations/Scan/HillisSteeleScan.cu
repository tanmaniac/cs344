/**
 *  Implementation of Hillis-Steele parallel prefix scan.
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

template <typename T>
__global__ void exclusiveScanKernel(T *dataOut, const T *dataIn, const size_t dataSize) {
    extern __shared__ T shmData[];

    int thId = threadIdx.x;
    int bufA = 0, bufB = 1;

    // Load everything into shared memory. We need to copy twice to fill the shared memory space
    shmData[bufA * dataSize + thId] = (thId == 0) ? 0 : dataIn[thId - 1];
    shmData[bufB * dataSize + thId] = (thId == 0) ? 0 : dataIn[thId - 1];
    __syncthreads();

    for (int offset = 1; offset < dataSize; offset <<= 1) {
        // Swap which side of the buffer we're writing into
        bufA = 1 - bufA;
        bufB = 1 - bufA;
        // Do scan step
        if (thId >= offset) {
            shmData[bufA * dataSize + thId] = shmData[bufB * dataSize + thId] + 
                                                    shmData[bufB * dataSize + thId - offset];
        } else {
            shmData[bufA * dataSize + thId] = shmData[bufB * dataSize + thId];
        }

        __syncthreads();
    }
    // Write to output array
    dataOut[thId] = shmData[bufA * dataSize + thId];
}

//template <typename T>
//void launchScanKernel(T *h_dataOut, const T *h_dataIn, const size_t dataSize) {
void launchScanKernel(int *h_dataOut, const int *h_dataIn, const size_t dataSize) {
    const size_t dataBytes = dataSize * sizeof(int);

    // Declare GPU memory pointers
    int *d_dataOut, *d_dataIn;

    // Allocate GPU memory
    cudaMalloc((void **) &d_dataOut, dataBytes);
    cudaMalloc((void **) &d_dataIn, dataBytes);

    cudaMemcpy(d_dataIn, h_dataIn, dataBytes, cudaMemcpyHostToDevice);

    exclusiveScanKernel<<<1, dataSize, 2 * dataBytes>>>(d_dataOut, d_dataIn, dataSize);

    // Copy back from GPU to CPU
    cudaMemcpy(h_dataOut, d_dataOut, dataBytes, cudaMemcpyDeviceToHost);

    // Free memory
    cudaFree(d_dataIn);
    cudaFree(d_dataOut);
}

//template void launchScanKernel<int>(int *h_dataOut, const int *h_dataIn, const size_t dataSize);