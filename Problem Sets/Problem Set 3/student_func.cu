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

__device__ inline int getPosition() {
    return blockIdx.x * blockDim.x + threadIdx.x;
}

__device__ inline boundsClamp(const int threadPos, const int offset, const size_t size) {
    return threadPos + offset >= size ? threadPos : threadPos + offset;
}

template <typename T>
__global__ void minMaxKernel(T* d_in, size_t dataSize, T* d_out) {
  static_assert(std::is_arithmetic<T>::value, "Only arithmetic types are supported");
    const int threadPos = getPosition();
    const int threadId = threadIdx.x;

    // Reduce this block
    for (unsigned int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadId < s) {
            // Make sure we don't exceed the bounds of the array. If this thread's comparison index
            // is outside the array bounds, just compare against itself.
            int compareIndex = threadPos + s >= size ? threadPos : threadPos + offset;
            d_in[threadPos].x = min(d_in[threadPos].x, d_in[compareIndex].x);
            d_in[threadPos].y = max(d_in[threadPos].y, d_in[compareIndex].y);
        }
        __syncthreads();
    }

    // Only write to output array with thread 0
    if (threadId == 0) {
        d_out[blockIdx.x] = d_in[threadPos];
    }
}

void launchMinMaxKernel(float* d_logLuminance, float& min_logLum, float& max_logLum, const size_t dataSize) {
    const size_t dataBytes = dataSize * sizeof (float2);

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
    const int maxThreadsPerBlock = 1024;
    const size_t dataSize = numRows * numCols;
    int threads = maxThreadsPerBlock;
    int blocks = dataSize / maxThreadsPerBlock;

    // Using float2 for shared memory so we can interleave min and max values
    const size_t shmSize = dataSize * sizeof(float2);
    minMaxKernel<<<blocks, threads, shmSize>>>(d)


    // 2) subtract them to find the range
    /*3) generate a histogram of all the values in the logLuminance channel using
         the formula: bin = (lum[i] - lumMin) / lumRange * numBins
    */
    /*4) Perform an exclusive scan (prefix sum) on the histogram to get
         the cumulative distribution of luminance values (this should go in the
         incoming d_cdf pointer which already has been allocated for you)
    */
}
