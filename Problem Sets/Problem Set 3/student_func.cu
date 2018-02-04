/* Udacity Homework 3
   HDR Tone-mapping

  Background HDR
  ==============

  A High Dynamic Range (HDR) image contains a wider variation of intensity
  and color than is allowed by the RGB format with 1 byte per channel that we
  have used in the previous assignment.

  To store this extra information we use single precision floating point for
  each channel.  This allows for an extremely wide range of intensity values.

  In the image for this assignment, the inside of church with light coming in
  through stained glass windows, the raw input floating point values for the
  channels range from 0 to 275.  But the mean is .41 and 98% of the values are
  less than 3!  This means that certain areas (the windows) are extremely bright
  compared to everywhere else.  If we linearly map this [0-275] range into the
  [0-255] range that we have been using then most values will be mapped to zero!
  The only thing we will be able to see are the very brightest areas - the
  windows - everything else will appear pitch black.

  The problem is that although we have cameras capable of recording the wide
  range of intensity that exists in the real world our monitors are not capable
  of displaying them.  Our eyes are also quite capable of observing a much wider
  range of intensities than our image formats / monitors are capable of
  displaying.

  Tone-mapping is a process that transforms the intensities in the image so that
  the brightest values aren't nearly so far away from the mean.  That way when
  we transform the values into [0-255] we can actually see the entire image.
  There are many ways to perform this process and it is as much an art as a
  science - there is no single "right" answer.  In this homework we will
  implement one possible technique.

  Background Chrominance-Luminance
  ================================

  The RGB space that we have been using to represent images can be thought of as
  one possible set of axes spanning a three dimensional space of color.  We
  sometimes choose other axes to represent this space because they make certain
  operations more convenient.

  Another possible way of representing a color image is to separate the color
  information (chromaticity) from the brightness information.  There are
  multiple different methods for doing this - a common one during the analog
  television days was known as Chrominance-Luminance or YUV.

  We choose to represent the image in this way so that we can remap only the
  intensity channel and then recombine the new intensity values with the color
  information to form the final image.

  Old TV signals used to be transmitted in this way so that black & white
  televisions could display the luminance channel while color televisions would
  display all three of the channels.


  Tone-mapping
  ============

  In this assignment we are going to transform the luminance channel (actually
  the log of the luminance, but this is unimportant for the parts of the
  algorithm that you will be implementing) by compressing its range to [0, 1].
  To do this we need the cumulative distribution of the luminance values.

  Example
  -------

  input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
  min / max / range: 0 / 9 / 9

  histo with 3 bins: [4 7 3]

  cdf : [4 11 14]


  Your task is to calculate this cumulative distribution by following these
  steps.

*/

#include "utils.h"

#include <array>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <iostream>

// Helper class to make generic 2-vector types
// https://stackoverflow.com/a/17834484
template <typename T, int cn>
struct MakeVec;
template <>
struct MakeVec<float, 2> {
    typedef float2 type;
};

__device__ inline int getPosition() {
    return blockIdx.x * blockDim.x + threadIdx.x;
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
__global__ void
    minMaxBlockReduceKernel(const T* d_in, const size_t dataSize, T* d_mins, T* d_maxes) {
    static_assert(std::is_arithmetic<T>::value, "Only arithmetic types are supported");
    const int threadPos = getPosition();
    const int threadId = threadIdx.x;

    // Shared memory is an array of 2-vector types so we can interleave min and max values
    typedef typename MakeVec<T, 2>::type vec2_type;
    extern __shared__ vec2_type minMaxVals[];

    // Copy values into shared memory. FLT_MAX is default value for min vals (everything is
    // lower) and -FLT_MAX is default for max vals (everything is higher)
    minMaxVals[threadId].x = threadPos < dataSize ? d_in[threadPos] : FLT_MAX;
    minMaxVals[threadId].y = threadPos < dataSize ? d_in[threadPos] : -FLT_MAX;
    __syncthreads();

    // Reduce this block
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadId < s) {
            minMaxVals[threadId].x = cuda_min(minMaxVals[threadId].x, minMaxVals[threadId + s].x);
            minMaxVals[threadId].y = cuda_max(minMaxVals[threadId].y, minMaxVals[threadId + s].y);
        }
        __syncthreads();
    }

    // Only write to output array with thread 0
    if (threadId == 0) {
        d_mins[blockIdx.x] = minMaxVals[0].x;
        d_maxes[blockIdx.x] = minMaxVals[0].y;
    }
}

template <typename T>
__global__ void minMaxWorkspaceKernel(T* d_mins, T* d_maxes, const size_t dataSize) {
    static_assert(std::is_arithmetic<T>::value, "Only arithmetic types are supported");
    const int threadPos = getPosition();
    const int threadId = threadIdx.x;

    // Define shared memory
    typedef typename MakeVec<T, 2>::type vec2_type;
    extern __shared__ vec2_type minMaxVals[];

    // Copy values into shared memory. FLT_MAX is default value for min vals (everything is
    // lower) and -FLT_MAX is default for max vals (everything is higher)
    minMaxVals[threadId].x = threadPos < dataSize ? d_mins[threadPos] : FLT_MAX;
    minMaxVals[threadId].y = threadPos < dataSize ? d_maxes[threadPos] : -FLT_MAX;
    __syncthreads();

    // Reduce this block
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadId < s) {
            minMaxVals[threadId].x = cuda_min(minMaxVals[threadId].x, minMaxVals[threadId + s].x);
            minMaxVals[threadId].y = cuda_max(minMaxVals[threadId].y, minMaxVals[threadId + s].y);
        }
        __syncthreads();
    }

    // Only write to output array with thread 0
    if (threadId == 0) {
        d_mins[0] = minMaxVals[0].x;
        d_maxes[0] = minMaxVals[0].y;
    }
}

template <typename T>
__global__ void globalHistoKernel(const T* const d_logLuminance,
                                  T min_logLum,
                                  T lumRange,
                                  const size_t dataSize,
                                  const size_t numBins,
                                  unsigned int* d_bins) {
    static_assert(std::is_arithmetic<T>::value, "Only arithmetic types are supported");
    const int threadPos = getPosition();
    // const int threadId = threadIdx.x;

    if (threadPos >= dataSize) {
        return;
    }

    // max_logLum will be at index 1024 in the cdf array:
    // (max_logLum - min_logLum) / lumRange * numBins
    // = lumRange / lumRange * numBins
    // = numBins
    // So just clamp the max index to numBins - 1
    unsigned int bin =
        cuda_min((unsigned int)((d_logLuminance[threadPos] - min_logLum) / lumRange * numBins),
                 (unsigned int)(numBins - 1));

    atomicAdd(&(d_bins[bin]), 1);
}

template <typename T>
__global__ void hillisSteeleScanKernel(const T* const d_in, const size_t dataSize, T* d_out) {
    extern __shared__ T shmData[];

    int threadId = threadIdx.x;
    int bufA = 0, bufB = 1;

    // Load everything into shared memory. We need to copy twice to fill the shared memory space
    shmData[bufA * dataSize + threadId] = (threadId == 0) ? 0 : d_in[threadId - 1];
    shmData[bufB * dataSize + threadId] = (threadId == 0) ? 0 : d_in[threadId - 1];
    __syncthreads();

    for (int offset = 1; offset < dataSize; offset <<= 1) {
        // Swap which side of the buffer we're writing into
        bufA = 1 - bufA;
        bufB = 1 - bufA;
        // Do scan step
        if (threadId >= offset) {
            shmData[bufA * dataSize + threadId] =
                shmData[bufB * dataSize + threadId] + shmData[bufB * dataSize + threadId - offset];
        } else {
            shmData[bufA * dataSize + threadId] = shmData[bufB * dataSize + threadId];
        }

        __syncthreads();
    }
    // Write to output array
    d_out[threadId] = shmData[bufA * dataSize + threadId];
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

void launchMinMaxKernel(const float* d_logLuminance,
                        float& min_logLum,
                        float& max_logLum,
                        const size_t numRows,
                        const size_t numCols) {

    const int maxThreadsPerBlock = 1024;
    size_t dataSize = numRows * numCols;
    int threads = maxThreadsPerBlock;
    // Determine how many blocks to run
    int blocks = dataSize / threads + 1;

    // Allocate enough memory to store min and max values from all the blocks
    const size_t minMaxArrLen = blocks;
    float *d_mins, *d_maxes;
    float h_mins[minMaxArrLen], h_maxes[minMaxArrLen];
    const size_t minMaxSize = minMaxArrLen * sizeof(float);
    checkCudaErrors(cudaMalloc(&d_mins, minMaxSize));
    checkCudaErrors(cudaMalloc(&d_maxes, minMaxSize));

    // Using two float arrays for shared memory so we can compute min and max simultaneously
    size_t shmSize = threads * sizeof(float2);
    minMaxBlockReduceKernel<<<blocks, threads, shmSize>>>(
        d_logLuminance, dataSize, d_mins, d_maxes);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());

    // Want a power of 2 for the final reduce step
    threads = roundUpToPow2(blocks);
    // Use the previous number of blocks as the new dataSize
    dataSize = blocks;
    blocks = 1;
    shmSize = threads * sizeof(float2);
    minMaxWorkspaceKernel<<<blocks, threads, shmSize>>>(d_mins, d_maxes, dataSize);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());

    checkCudaErrors(cudaMemcpy(h_mins, d_mins, dataSize * sizeof(float), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_maxes, d_maxes, dataSize * sizeof(float), cudaMemcpyDeviceToHost));
    min_logLum = h_mins[0];
    max_logLum = h_maxes[0];

    // Free memory
    checkCudaErrors(cudaFree(d_mins));
    checkCudaErrors(cudaFree(d_maxes));
}

void launchHistoKernel(const float* const d_logLuminance,
                       float min_logLum,
                       float lumRange,
                       const size_t dataSize,
                       const size_t numBins,
                       unsigned int* const d_bins) {
    const int maxThreadsPerBlock = 1024;
    const int threads = maxThreadsPerBlock;
    const int blocks = dataSize / threads + 1;

    globalHistoKernel<<<blocks, threads>>>(
        d_logLuminance, min_logLum, lumRange, dataSize, numBins, d_bins);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());
}

void launchScanKernel(const unsigned int* const d_bins,
                      const size_t numBins,
                      unsigned int* d_cdf) {
    // Shared memory size is double the size of the d_bins array since the kernel double buffers the
    // input array
    const size_t shmSize = numBins * sizeof(unsigned int) * 2;
    // numBins is 1024, which is the maximum number of threads we can launch on a block, so we just
    // use one block with numBins threads
    hillisSteeleScanKernel<<<1, numBins, shmSize>>>(d_bins, numBins, d_cdf);
}

void your_histogram_and_prefixsum(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float& min_logLum,
                                  float& max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins) {
    // TODO
    /*Here are the steps you need to implement
      1) find the minimum and maximum value in the input logLuminance channel
         store in min_logLum and max_logLum
    */
    launchMinMaxKernel(d_logLuminance, min_logLum, max_logLum, numRows, numCols);

    /* 2) subtract them to find the range
    */
    float range = max_logLum - min_logLum;
    //std::cout << "Min = " << min_logLum << ", Max = " << max_logLum << ", Range = " << range
    //          << std::endl;

    /*3) generate a histogram of all the values in the logLuminance channel using
         the formula: bin = (lum[i] - lumMin) / lumRange * numBins
    */
    //std::cout << "NumBins = " << numBins << std::endl;
    unsigned int* d_bins;
    checkCudaErrors(cudaMalloc(&d_bins, numBins * sizeof(unsigned int)));
    checkCudaErrors(cudaMemset(d_bins, 0, numBins * sizeof(unsigned int)));
    launchHistoKernel(d_logLuminance, min_logLum, range, numRows * numCols, numBins, d_bins);

    std::array<unsigned int, 1024> histo;
    checkCudaErrors(cudaMemcpy(histo.data(), d_bins, numBins * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    /*4) Perform an exclusive scan (prefix sum) on the histogram to get
         the cumulative distribution of luminance values (this should go in the
         incoming d_cdf pointer which already has been allocated for you)
    */
    launchScanKernel(d_bins, numBins, d_cdf);
}
