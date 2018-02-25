// Udacity HW 6
// Poisson Blending

/* Background
   ==========

   The goal for this assignment is to take one image (the source) and
   paste it into another image (the destination) attempting to match the
   two images so that the pasting is non-obvious. This is
   known as a "seamless clone".

   The basic ideas are as follows:

   1) Figure out the interior and border of the source image
   2) Use the values of the border pixels in the destination image
      as boundary conditions for solving a Poisson equation that tells
      us how to blend the images.

      No pixels from the destination except pixels on the border
      are used to compute the match.

   Solving the Poisson Equation
   ============================

   There are multiple ways to solve this equation - we choose an iterative
   method - specifically the Jacobi method. Iterative methods start with
   a guess of the solution and then iterate to try and improve the guess
   until it stops changing.  If the problem was well-suited for the method
   then it will stop and where it stops will be the solution.

   The Jacobi method is the simplest iterative method and converges slowly -
   that is we need a lot of iterations to get to the answer, but it is the
   easiest method to write.

   Jacobi Iterations
   =================

   Our initial guess is going to be the source image itself.  This is a pretty
   good guess for what the blended image will look like and it means that
   we won't have to do as many iterations compared to if we had started far
   from the final solution.

   ImageGuess_prev (Floating point)
   ImageGuess_next (Floating point)

   DestinationImg
   SourceImg

   Follow these steps to implement one iteration:

   1) For every pixel p in the interior, compute two sums over the four neighboring pixels:
      Sum1: If the neighbor is in the interior then += ImageGuess_prev[neighbor]
             else if the neighbor in on the border then += DestinationImg[neighbor]

      Sum2: += SourceImg[p] - SourceImg[neighbor]   (for all four neighbors)

   2) Calculate the new pixel value:
      float newVal= (Sum1 + Sum2) / 4.f  <------ Notice that the result is FLOATING POINT
      ImageGuess_next[p] = min(255, max(0, newVal)); //clamp to [0, 255]


    In this assignment we will do 800 iterations.
   */

#include "utils.h"

#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/pair.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/transform_reduce.h>
#include <opencv2/highgui/highgui.hpp>

#include <iostream>

namespace Blend {
enum Channels { RED, GREEN, BLUE };

typedef unsigned int MaskRegions;
enum { UNMASKED = 0, BORDER = 1, INTERIOR = 2 };
}; // namespace Blend

__device__ inline uint2 getPosition() {
    return make_uint2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
}

__device__ inline unsigned int convert2dTo1d(const uint2 loc, const size_t numCols) {
    return loc.y * numCols + loc.x;
}

// Adapted from Thrust's bounding_box example
// https://github.com/thrust/thrust/blob/master/examples/bounding_box.cu
struct Point2D {
    unsigned int _x, _y;

    // Initialize point with (x, y) coords of 0
    __host__ __device__ Point2D() : _x(0), _y(0) {}

    __host__ __device__ Point2D(unsigned int x, unsigned int y) : _x(x), _y(y) {}

    __host__ __device__ Point2D(const Point2D& point) : _x(point._x), _y(point._y) {}
};

struct BoundingBox {
    Point2D _upperLeft, _lowerRight;

    __host__ __device__ BoundingBox() {}

    __host__ __device__ BoundingBox(const Point2D& point) : _upperLeft(point), _lowerRight(point) {}

    __host__ __device__ BoundingBox(const Point2D& upperLeft, const Point2D& lowerRight)
        : _upperLeft(upperLeft), _lowerRight(lowerRight) {}

    __host__ __device__ size_t size() {
        return (_lowerRight._x - _upperLeft._x) * (_lowerRight._y - _upperLeft._y);
    }

    __host__ __device__ size_t numRows() {
        return _lowerRight._y - _upperLeft._y;
    }

    __host__ __device__ size_t numCols() {
        return _lowerRight._x - _upperLeft._x;
    }
};

struct BBoxReduction : public thrust::binary_function<BoundingBox, BoundingBox, BoundingBox> {
    __host__ __device__ BoundingBox operator()(BoundingBox a, BoundingBox b) {
        // Lower left corner
        Point2D upperLeft(thrust::min(a._upperLeft._x, b._upperLeft._x),
                          thrust::min(a._upperLeft._y, b._upperLeft._y));
        // Upper right corner
        Point2D lowerRight(thrust::max(a._lowerRight._x, b._lowerRight._x),
                           thrust::max(a._lowerRight._y, b._lowerRight._y));

        return BoundingBox(upperLeft, lowerRight);
    }
};

__global__ void maskKernel(const uchar4* const sourceImg,
                           const size_t numRows,
                           const size_t numCols,
                           Blend::MaskRegions* mask) {
    const uint2 threadPos = getPosition();
    // Don't exceed the edges of the input image
    if (threadPos.x >= numCols || threadPos.y >= numRows) {
        return;
    }

    const unsigned int threadLoc = convert2dTo1d(threadPos, numCols);
    uchar4 pixel = sourceImg[threadLoc];
    mask[threadLoc] =
        (pixel.x == 255 && pixel.y == 255 && pixel.z == 255) ? Blend::UNMASKED : Blend::BORDER;
}

// Compact the masked points into an array of Point2Ds
__global__ void compactMaskToPoint2D(const Blend::MaskRegions* const mask,
                                     const unsigned int* const maskScan,
                                     const size_t numRows,
                                     const size_t numCols,
                                     Point2D* maskedPoints) {
    const uint2 threadPos = getPosition();
    // Don't exceed the edges of the input image
    if (threadPos.x >= numCols || threadPos.y >= numRows) {
        return;
    }

    const unsigned int threadLoc = convert2dTo1d(threadPos, numCols);
    if (mask[threadLoc] == Blend::BORDER) {
        maskedPoints[maskScan[threadLoc]] = Point2D(threadLoc % numCols, threadLoc / numCols);
    }
}

// Take an image or mask and crop it given the upper left and lower right bounds
template <typename T>
__global__ void crop(const T* const source,
                     const size_t numColsSrc,
                     const Point2D upperLeft,
                     const Point2D lowerRight,
                     T* cropped) {
    const uint2 threadPos = getPosition();
    // Snap to bounding box
    if (threadPos.x >= lowerRight._x || threadPos.y >= lowerRight._y ||
        threadPos.x < upperLeft._x || threadPos.y < upperLeft._y) {
        return;
    }

    const unsigned int threadLoc = convert2dTo1d(threadPos, numColsSrc);
    // Upper left coordinate of the bounding box is the new (0, 0)
    const uint2 croppedThreadPos =
        make_uint2(threadPos.x - upperLeft._x, threadPos.y - upperLeft._y);
    const unsigned int croppedThreadLoc =
        convert2dTo1d(croppedThreadPos, lowerRight._x - upperLeft._x);

    cropped[croppedThreadLoc] = source[threadLoc];
}

// TODO: use shared memory instead of global memory
__global__ void markInterior(Blend::MaskRegions* mask, const size_t numRows, const size_t numCols) {
    const uint2 threadPos = getPosition();
    // Don't exceed the edges of the input image
    if (threadPos.x >= numCols || threadPos.y >= numRows) {
        return;
    }

    const unsigned int threadLoc = convert2dTo1d(threadPos, numCols);
    // return if this isn't a masked point anyway
    if (mask[threadLoc] == Blend::UNMASKED) {
        return;
    }

    // Iterate over the neighbors and determine if all of the neighbors are inside the mask
    uint2 neighbors[4] = {make_uint2(threadPos.x - 1, threadPos.y),
                          make_uint2(threadPos.x, threadPos.y - 1),
                          make_uint2(threadPos.x + 1, threadPos.y),
                          make_uint2(threadPos.x, threadPos.y + 1)};
    unsigned int neighborMasks = 0;
    unsigned int neighborLoc;
    for (const auto& nbr : neighbors) {
        // Don't need to check if less than 0 because subtraction will just overflow to 2^32 - 1
        if (nbr.x < numCols && nbr.y < numRows) {
            neighborLoc = convert2dTo1d(nbr, numCols);
            if (mask[neighborLoc] != Blend::UNMASKED) {
                neighborMasks++;
            }
        }
    }
    // If all the neighbors are inside the mask, set this pixel to an interior pixel
    if (neighborMasks == 4) {
        mask[threadLoc] = Blend::INTERIOR;
    }
}

__global__ void extractColor(const uchar4* const pixels,
                             Blend::Channels whichChannel,
                             const size_t numRows,
                             const size_t numCols,
                             unsigned char* channel) {
    const uint2 threadPos = getPosition();

    if (threadPos.x >= numCols || threadPos.y >= numRows) {
        return;
    }
    const unsigned int threadLoc = convert2dTo1d(threadPos, numCols);

    switch (whichChannel) {
    case Blend::RED:
        channel[threadLoc] = pixels[threadLoc].x;
        break;
    case Blend::GREEN:
        channel[threadLoc] = pixels[threadLoc].y;
        break;
    case Blend::BLUE:
        channel[threadLoc] = pixels[threadLoc].z;
        break;
    }
}

// Initial imgGuessPrev values that are used by the Jacobi Solver kernel
__global__ void precomputeChannelValues(float* imgGuessPrev,
                                        const Blend::MaskRegions* const mask,
                                        const size_t numRows,
                                        const size_t numCols) {
    const uint2 threadPos = getPosition();
    // Don't exceed the edges of the input image
    if (threadPos.x >= numCols || threadPos.y >= numRows) {
        return;
    }

    const unsigned int threadLoc = convert2dTo1d(threadPos, numCols);

    if (mask[threadLoc] == Blend::INTERIOR) {
        float sum = 4.f * imgGuessPrev[threadLoc];
        sum -= imgGuessPrev[threadLoc - 1] + imgGuessPrev[threadLoc + 1];
        sum -= imgGuessPrev[threadLoc - numCols] + imgGuessPrev[threadLoc + numCols];
        imgGuessPrev[threadLoc] = sum;
    }
}

// TODO: Use shared memory instead of global memory
/* Our initial guess is going to be the source image itself.  This is a pretty
    good guess for what the blended image will look like and it means that
    we won't have to do as many iterations compared to if we had started far
    from the final solution.

    ImageGuess_prev (Floating point)
    ImageGuess_next (Floating point)

    DestinationImg
    SourceImg

    Follow these steps to implement one iteration:

    1) For every pixel p in the interior, compute two sums over the four neighboring pixels:
        Sum1: If the neighbor is in the interior then += ImageGuess_prev[neighbor]
              else if the neighbor in on the border then += DestinationImg[neighbor]

        Sum2: += SourceImg[p] - SourceImg[neighbor]   (for all four neighbors)

    2) Calculate the new pixel value:
        float newVal= (Sum1 + Sum2) / 4.f  <------ Notice that the result is FLOATING POINT
        ImageGuess_next[p] = min(255, max(0, newVal)); //clamp to [0, 255]
*/
__global__ void jacobiSolver(float* imgGuessPrev,
                             float* imgGuessNext,
                             const Blend::MaskRegions* const mask,
                             const unsigned char* const source,
                             const unsigned char* const dest,
                             const size_t numRows,
                             const size_t numCols) {
    const uint2 threadPos = getPosition();
    // Don't exceed the edges of the input image
    if (threadPos.x >= numCols || threadPos.y >= numRows) {
        return;
    }

    const unsigned int threadLoc = convert2dTo1d(threadPos, numCols);
    // return if this isn't a masked point anyway
    if (mask[threadLoc] != Blend::INTERIOR) {
        return;
    }

    // Find neighbors
    uint2 neighbors[4] = {make_uint2(threadPos.x - 1, threadPos.y),
                          make_uint2(threadPos.x, threadPos.y - 1),
                          make_uint2(threadPos.x + 1, threadPos.y),
                          make_uint2(threadPos.x, threadPos.y + 1)};
    float sum1 = 0.f;
    float sum2 = 0.f;

    for (const auto& nbr : neighbors) {
        unsigned int nbrLoc = convert2dTo1d(nbr, numCols);
        sum1 += float((mask[nbrLoc] == Blend::INTERIOR) ? imgGuessPrev[nbrLoc] : dest[nbrLoc]);
        sum2 += float(source[threadLoc] - source[nbrLoc]);
    }
    float newVal = (sum1 + sum2) / 4.f;
    // float newVal = (sum1 + imgGuessNext[threadLoc]) / 4.f;
    // imgGuessPrev[threadLoc] = imgGuessNext[threadLoc];
    imgGuessNext[threadLoc] = min(255.f, max(0.f, newVal));
}

// Inputs are the RGB channels of the cropped area to paste into the destination image, along with
// the bounding box from which the crop was taken. Grid size and block size are assumed to be
// computed for the cropped area and NOT for the destination image
__global__ void copyChannelsToDest(const float* const red,
                                   const float* const green,
                                   const float* const blue,
                                   const Blend::MaskRegions* const mask,
                                   const size_t numRows,
                                   const size_t numCols,
                                   const BoundingBox pasteTo,
                                   const size_t destNumCols,
                                   uchar4* dest) {
    const uint2 threadPos = getPosition();

    if (threadPos.x >= numCols || threadPos.y >= numRows) {
        return;
    }

    const unsigned int threadLoc = convert2dTo1d(threadPos, numCols);
    // Only copy interior pixels
    if (mask[threadLoc] != Blend::INTERIOR) {
        return;
    }

    // Position of this pixel in relation to the (0, 0) point of the destination image
    const uint2 destThreadPos =
        make_uint2(threadPos.x + pasteTo._upperLeft._x, threadPos.y + pasteTo._upperLeft._y);
    const unsigned int destThreadLoc = convert2dTo1d(destThreadPos, destNumCols);

    // build pixel out of the input channels
    uchar4 pixel = make_uchar4(red[threadLoc], green[threadLoc], blue[threadLoc], 255);
    // Paste into output
    dest[destThreadLoc] = pixel;
}

__global__ void buildMaskedImage(const uchar4* const pixels,
                                 const Blend::MaskRegions* const mask,
                                 const size_t numRows,
                                 const size_t numCols,
                                 uchar4* out) {
    const uint2 threadPos = getPosition();

    if (threadPos.x >= numCols || threadPos.y >= numRows) {
        return;
    }
    const unsigned int threadLoc = convert2dTo1d(threadPos, numCols);
    if (mask[threadLoc] == Blend::UNMASKED) {
        out[threadLoc] = make_uchar4(0, 0, 0, 255);
    } else {
        out[threadLoc] = pixels[threadLoc];
    }
}

// Builds an image representation of the mask, where:
//  Blend::UNMASKED = not part of the mask -> black
//  Blend::BORDER   = border of the mask -> blue
//  Blend::INTERIOR = interior of the mask -> green
__global__ void buildInteriorImage(const Blend::MaskRegions* const mask,
                                   const size_t numRows,
                                   const size_t numCols,
                                   uchar4* out) {
    const uint2 threadPos = getPosition();

    if (threadPos.x >= numCols || threadPos.y >= numRows) {
        return;
    }
    const unsigned int threadLoc = convert2dTo1d(threadPos, numCols);
    if (mask[threadLoc] == Blend::INTERIOR) {
        out[threadLoc] = make_uchar4(0, 255, 0, 255);
    } else if (mask[threadLoc] == Blend::BORDER) {
        out[threadLoc] = make_uchar4(255, 0, 0, 255);
    } else {
        out[threadLoc] = make_uchar4(0, 0, 0, 255);
    }
}

// Transform the mask created in step 1 of your_blend into a 1D array of points, where each point is
// a masked pixel
void findBoundingBox(const thrust::device_vector<Blend::MaskRegions>& mask,
                     const size_t numRows,
                     const size_t numCols,
                     dim3 blocks,
                     dim3 threads,
                     BoundingBox& result) {
    const size_t imageSize = numRows * numCols;
    thrust::device_vector<unsigned int> maskScan(imageSize);

    // Scan the mask vector to get the output indices
    thrust::exclusive_scan(mask.begin(), mask.end(), maskScan.begin());

    // Figure out how many indices we need. This will just be the max element of the scan, which
    // will be one greater than the number of points in the mask
    thrust::device_vector<unsigned int>::iterator maxIter =
        thrust::max_element(maskScan.begin(), maskScan.end());
    if (maxIter == maskScan.end()) {
        std::cerr << "Couldn't find number of masked points" << std::endl;
        exit(1);
    }

    std::cout << "Mask is " << *maxIter << " indices long" << std::endl;

    // Vector to hold the Point2Ds
    thrust::device_vector<Point2D> maskedPoints(*maxIter);
    thrust::fill(maskedPoints.begin(), maskedPoints.end(), Point2D(0, 0));
    // Compact the full mask into just the points we need, while converting it from an array of mask
    // values into an array of 2D points at which those mask values are placed
    compactMaskToPoint2D<<<blocks, threads>>>(
        mask.data().get(), maskScan.data().get(), numRows, numCols, maskedPoints.data().get());

    // Compute bounding box of the mask
    // Initial bounding box just contains the first point
    BoundingBox init(maskedPoints[0], maskedPoints[0]);
    // Binary reduction operation
    BBoxReduction binaryOp;
    result =
        thrust::reduce(thrust::device, maskedPoints.begin(), maskedPoints.end(), init, binaryOp);
    // Extend the bounding box by one pixel so that the right and bottom edges are non-inclusive,
    // like a normal image
    result._lowerRight._x += 1;
    result._lowerRight._y += 1;

    std::cout << "Bounding box: top left = (" << result._upperLeft._x << ", "
              << result._upperLeft._y << "); bottom right = (" << result._lowerRight._x << ", "
              << result._lowerRight._y << ")" << std::endl;
}

void drawImageFromGPU(const thrust::device_vector<uchar4>& img,
                      const size_t numRows,
                      const size_t numCols,
                      const std::string& title) {
    std::vector<cv::Vec4b> hostImage;
    hostImage.resize(img.size());
    for (unsigned int i = 0; i < img.size(); i++) {
        hostImage[i] = cv::Vec<uchar, 4>(static_cast<uchar4>(img[i]).x,
                                         static_cast<uchar4>(img[i]).y,
                                         static_cast<uchar4>(img[i]).z,
                                         static_cast<uchar4>(img[i]).w);
    }
    const cv::Mat imgMat(cv::Size(numCols, numRows), CV_8UC4, hostImage.data());

    cv::namedWindow(title);
    cv::imwrite(title + ".png", imgMat);
#ifdef DRAW_DEBUG_IMGS
    cv::imshow(title, imgMat);
    cv::waitKey(0);
#endif
}

void your_blend(const uchar4* const h_sourceImg, // IN
                const size_t numRowsSource,
                const size_t numColsSource,
                const uchar4* const h_destImg, // IN
                uchar4* const h_blendedImg)    // OUT
{
    static constexpr size_t TILE_SIZE = 8;
    static constexpr size_t CROP_PAD_SIZE = 2;
    /* To Recap here are the steps you need to implement

       1) Compute a mask of the pixels from the source image to be copied
          The pixels that shouldn't be copied are completely white, they
          have R=255, G=255, B=255.  Any other pixels SHOULD be copied. */

    // Allocate device memory
    const size_t sourceLength = numRowsSource * numColsSource;
    std::cout << "sourceLength = " << sourceLength << std::endl;
    thrust::device_vector<uchar4> sourceImg(sourceLength);
    thrust::device_vector<uchar4> destImg(sourceLength);
    // Copy from host to device
    thrust::copy(h_sourceImg, h_sourceImg + sourceLength, sourceImg.begin());
    thrust::copy(h_destImg, h_destImg + sourceLength, destImg.begin());

    // Vector to hold the mask
    thrust::device_vector<Blend::MaskRegions> mask(sourceLength);

    dim3 blocks(max(1, (unsigned int)ceil(float(numColsSource) / float(TILE_SIZE))),
                max(1, (unsigned int)ceil(float(numRowsSource) / float(TILE_SIZE))));
    dim3 threads(TILE_SIZE, TILE_SIZE);

    maskKernel<<<blocks, threads>>>(
        sourceImg.data().get(), numRowsSource, numColsSource, mask.data().get());

#ifdef SAVE_DEBUG_IMGS
    // Draw the masked image
    thrust::device_vector<uchar4> maskedImg(sourceLength);
    buildMaskedImage<<<blocks, threads>>>(sourceImg.data().get(),
                                          mask.data().get(),
                                          numRowsSource,
                                          numColsSource,
                                          maskedImg.data().get());

    drawImageFromGPU(maskedImg, numRowsSource, numColsSource, "Masked image");
#endif

    // Compute an axis-aligned bounding box so that we can only operate on the pixels we need to,
    // instead of on the entire image
    std::cout << "Finding bounding box" << std::endl;
    BoundingBox bbox;
    findBoundingBox(mask, numRowsSource, numColsSource, blocks, threads, bbox);
    // Extend the bounding box by a little bit
    bbox._upperLeft =
        Point2D(bbox._upperLeft._x - CROP_PAD_SIZE, bbox._upperLeft._y - CROP_PAD_SIZE);
    bbox._lowerRight =
        Point2D(bbox._lowerRight._x + CROP_PAD_SIZE, bbox._lowerRight._y + CROP_PAD_SIZE);

    // Make a smaller image with the dimensions of the bounding box, and copy those pixels from the
    // source image into it
    const size_t numCroppedRows = bbox.numRows();
    const size_t numCroppedCols = bbox.numCols();
    const size_t croppedSize = bbox.size();
    thrust::device_vector<uchar4> croppedSrcImg(croppedSize);
    thrust::device_vector<uchar4> croppedDestImg(croppedSize);
    std::cout << "cropping with " << numCroppedRows << " rows, " << numCroppedCols << " cols, "
              << croppedSrcImg.size() << " total size" << std::endl;
    crop<<<blocks, threads>>>(sourceImg.data().get(),
                              numColsSource,
                              bbox._upperLeft,
                              bbox._lowerRight,
                              croppedSrcImg.data().get());
    crop<<<blocks, threads>>>(destImg.data().get(),
                              numColsSource,
                              bbox._upperLeft,
                              bbox._lowerRight,
                              croppedDestImg.data().get());

#ifdef SAVE_DEBUG_IMGS
    drawImageFromGPU(croppedSrcImg, numCroppedRows, numCroppedCols, "Cropped image");
    drawImageFromGPU(croppedDestImg, numCroppedRows, numCroppedCols, "Cropped destination image");
#endif

    // Crop the mask down to match the cropped source image
    thrust::device_vector<Blend::MaskRegions> croppedMask(croppedSize);
    crop<<<blocks, threads>>>(mask.data().get(),
                              numColsSource,
                              bbox._upperLeft,
                              bbox._lowerRight,
                              croppedMask.data().get());

    /*  2) Compute the interior and border regions of the mask.  An interior
          pixel has all 4 neighbors also inside the mask.  A border pixel is
          in the mask itself, but has at least one neighbor that isn't.*/
    blocks = dim3(max(1, (unsigned int)ceil(float(numCroppedCols) / float(TILE_SIZE))),
                  max(1, (unsigned int)ceil(float(numCroppedRows) / float(TILE_SIZE))));

    markInterior<<<blocks, threads>>>(croppedMask.data().get(), numCroppedRows, numCroppedCols);
    cudaDeviceSynchronize();

#ifdef SAVE_DEBUG_IMGS
    // Draw interior image
    thrust::device_vector<uchar4> interiorImage(croppedSize);
    buildInteriorImage<<<blocks, threads>>>(
        croppedMask.data().get(), numCroppedRows, numCroppedCols, interiorImage.data().get());
    drawImageFromGPU(interiorImage, numCroppedRows, numCroppedCols, "Interior of mask");
#endif

    /* 3) Separate out the incoming image into three separate channels */
    thrust::device_vector<unsigned char> croppedSrcRed(croppedSize);
    thrust::device_vector<unsigned char> croppedSrcGreen(croppedSize);
    thrust::device_vector<unsigned char> croppedSrcBlue(croppedSize);

    thrust::device_vector<unsigned char> croppedDestRed(croppedSize);
    thrust::device_vector<unsigned char> croppedDestGreen(croppedSize);
    thrust::device_vector<unsigned char> croppedDestBlue(croppedSize);

    // Source image channel extraction
    extractColor<<<blocks, threads>>>(croppedSrcImg.data().get(),
                                      Blend::RED,
                                      numCroppedRows,
                                      numCroppedCols,
                                      croppedSrcRed.data().get());
    extractColor<<<blocks, threads>>>(croppedSrcImg.data().get(),
                                      Blend::GREEN,
                                      numCroppedRows,
                                      numCroppedCols,
                                      croppedSrcGreen.data().get());
    extractColor<<<blocks, threads>>>(croppedSrcImg.data().get(),
                                      Blend::BLUE,
                                      numCroppedRows,
                                      numCroppedCols,
                                      croppedSrcBlue.data().get());

    // Destination image channel extraction
    extractColor<<<blocks, threads>>>(croppedDestImg.data().get(),
                                      Blend::RED,
                                      numCroppedRows,
                                      numCroppedCols,
                                      croppedDestRed.data().get());
    extractColor<<<blocks, threads>>>(croppedDestImg.data().get(),
                                      Blend::GREEN,
                                      numCroppedRows,
                                      numCroppedCols,
                                      croppedDestGreen.data().get());
    extractColor<<<blocks, threads>>>(croppedDestImg.data().get(),
                                      Blend::BLUE,
                                      numCroppedRows,
                                      numCroppedCols,
                                      croppedDestBlue.data().get());

    /* 4) Create two float(!) buffers for each color channel that will
          act as our guesses.  Initialize them to the respective color
          channel of the source image since that will act as our intial guess. */

    thrust::device_vector<float> guessRedPrev(croppedSize);
    thrust::device_vector<float> guessRedNext(croppedSize);
    thrust::device_vector<float> guessGreenPrev(croppedSize);
    thrust::device_vector<float> guessGreenNext(croppedSize);
    thrust::device_vector<float> guessBluePrev(croppedSize);
    thrust::device_vector<float> guessBlueNext(croppedSize);

    // Is there really no better way to do this?
    thrust::copy(croppedSrcRed.begin(), croppedSrcRed.end(), guessRedPrev.begin());
    thrust::copy(croppedSrcRed.begin(), croppedSrcRed.end(), guessRedNext.begin());
    thrust::copy(croppedSrcGreen.begin(), croppedSrcGreen.end(), guessGreenPrev.begin());
    thrust::copy(croppedSrcGreen.begin(), croppedSrcGreen.end(), guessGreenNext.begin());
    thrust::copy(croppedSrcBlue.begin(), croppedSrcBlue.end(), guessBluePrev.begin());
    thrust::copy(croppedSrcBlue.begin(), croppedSrcBlue.end(), guessBlueNext.begin());

    /* 5) For each color channel perform the Jacobi iteration described
          above 800 times. */

    static constexpr unsigned int NUM_ITERATIONS = 800;
    for (unsigned int i = 0; i < NUM_ITERATIONS; i++) {
        if (i % 2 == 0) {
            jacobiSolver<<<blocks, threads>>>(guessRedNext.data().get(),
                                              guessRedPrev.data().get(),
                                              croppedMask.data().get(),
                                              croppedSrcRed.data().get(),
                                              croppedDestRed.data().get(),
                                              numCroppedRows,
                                              numCroppedCols);
            jacobiSolver<<<blocks, threads>>>(guessGreenNext.data().get(),
                                              guessGreenPrev.data().get(),
                                              croppedMask.data().get(),
                                              croppedSrcGreen.data().get(),
                                              croppedDestGreen.data().get(),
                                              numCroppedRows,
                                              numCroppedCols);
            jacobiSolver<<<blocks, threads>>>(guessBlueNext.data().get(),
                                              guessBluePrev.data().get(),
                                              croppedMask.data().get(),
                                              croppedSrcBlue.data().get(),
                                              croppedDestBlue.data().get(),
                                              numCroppedRows,
                                              numCroppedCols);
        } else {

            jacobiSolver<<<blocks, threads>>>(guessRedPrev.data().get(),
                                              guessRedNext.data().get(),
                                              croppedMask.data().get(),
                                              croppedSrcRed.data().get(),
                                              croppedDestRed.data().get(),
                                              numCroppedRows,
                                              numCroppedCols);
            jacobiSolver<<<blocks, threads>>>(guessGreenPrev.data().get(),
                                              guessGreenNext.data().get(),
                                              croppedMask.data().get(),
                                              croppedSrcGreen.data().get(),
                                              croppedDestGreen.data().get(),
                                              numCroppedRows,
                                              numCroppedCols);
            jacobiSolver<<<blocks, threads>>>(guessBluePrev.data().get(),
                                              guessBlueNext.data().get(),
                                              croppedMask.data().get(),
                                              croppedSrcBlue.data().get(),
                                              croppedDestBlue.data().get(),
                                              numCroppedRows,
                                              numCroppedCols);
        }
        // cudaDeviceSynchronize();
    }

    /* 6) Create the output image by replacing all the interior pixels
          in the destination image with the result of the Jacobi iterations.
          Just cast the floating point values to unsigned chars since we have
          already made sure to clamp them to the correct range. */
    copyChannelsToDest<<<blocks, threads>>>(guessRedNext.data().get(),
                                            guessGreenNext.data().get(),
                                            guessBlueNext.data().get(),
                                            croppedMask.data().get(),
                                            numCroppedRows,
                                            numCroppedCols,
                                            bbox,
                                            numColsSource,
                                            destImg.data().get());
#ifdef SAVE_DEBUG_IMGS
    drawImageFromGPU(destImg, numRowsSource, numColsSource, "output");
#endif
    thrust::copy(destImg.begin(), destImg.end(), h_blendedImg);

    /*  Since this is final assignment we provide little boilerplate code to
        help you.  Notice that all the input/output pointers are HOST pointers.

        You will have to allocate all of your own GPU memory and perform your own
        memcopies to get data in and out of the GPU memory.

        Remember to wrap all of your calls with checkCudaErrors() to catch any
        thing that might go wrong.  After each kernel call do:

        cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

        to catch any errors that happened while executing the kernel.
    */

    // Wow, don't need to clean up any memory because I'm a T H R U S T B O Y E
}
