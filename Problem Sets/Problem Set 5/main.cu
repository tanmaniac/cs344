#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include "timer.h"
#include "utils.h"
#if defined(_WIN16) || defined(_WIN32) || defined(_WIN64)
#include <Windows.h>
#else
#include <sys/time.h>
#endif

#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/random/linear_congruential_engine.h>
#include <thrust/random/normal_distribution.h>
#include <thrust/random/uniform_int_distribution.h>

#include "reference_calc.h"

void computeHistogram(const unsigned int* const d_vals,
                      unsigned int* const d_histo,
                      const unsigned int numBins,
                      const unsigned int numElems,
                      const cudaDeviceProp& devProps);

void thrustHistogram(thrust::device_vector<unsigned int>& vals,
                     unsigned int* histo,
                     int numBins,
                     int numVals);

int main(void) {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        fprintf(stderr, "error: no devices supporting CUDA.\n");
        exit(EXIT_FAILURE);
    }
    int dev = 0;
    cudaSetDevice(dev);

    cudaDeviceProp devProps;
    if (cudaGetDeviceProperties(&devProps, dev) == 0) {
        printf("Using device %d:\n", dev);
        printf("%s; global mem: %dB; compute v%d.%d; clock: %d kHz\n",
               devProps.name,
               (int)devProps.totalGlobalMem,
               (int)devProps.major,
               (int)devProps.minor,
               (int)devProps.clockRate);
    }

    const unsigned int numBins = 1024;
    const unsigned int numElems = 10000 * numBins;
    const float stddev = 100.f;

    unsigned int* vals = new unsigned int[numElems];
    unsigned int* h_vals = new unsigned int[numElems];
    unsigned int* h_studentHisto = new unsigned int[numBins];
    unsigned int* h_refHisto = new unsigned int[numBins];

#if defined(_WIN16) || defined(_WIN32) || defined(_WIN64)
    srand(GetTickCount());
#else
    timeval tv;
    gettimeofday(&tv, NULL);

    srand(tv.tv_usec);
#endif

    // make the mean unpredictable, but close enough to the middle
    // so that timings are unaffected
    unsigned int mean = rand() % 100 + 462;

    // Output mean so that grading can happen with the same inputs
    std::cout << mean << std::endl;

    thrust::minstd_rand rng;

    thrust::random::normal_distribution<float> normalDist((float)mean, stddev);

    // Generate the random values
    for (size_t i = 0; i < numElems; ++i) {
        vals[i] = std::min((unsigned int)std::max((int)normalDist(rng), 0), numBins - 1);
    }

    unsigned int *d_vals, *d_histo;

    GpuTimer timer;

    checkCudaErrors(cudaMalloc(&d_vals, sizeof(unsigned int) * numElems));
    checkCudaErrors(cudaMalloc(&d_histo, sizeof(unsigned int) * numBins));
    checkCudaErrors(cudaMemset(d_histo, 0, sizeof(unsigned int) * numBins));

    checkCudaErrors(
        cudaMemcpy(d_vals, vals, sizeof(unsigned int) * numElems, cudaMemcpyHostToDevice));

    timer.Start();
    computeHistogram(d_vals, d_histo, numBins, numElems, devProps);
    timer.Stop();
    int err = printf("Your code ran in: %f msecs.\n", timer.Elapsed());

    if (err < 0) {
        // Couldn't print! Probably the student closed stdout - bad news
        std::cerr << "Couldn't print timing information! STDOUT Closed!" << std::endl;
        exit(1);
    }

    // copy the student-computed histogram back to the host
    checkCudaErrors(cudaMemcpy(
        h_studentHisto, d_histo, sizeof(unsigned int) * numBins, cudaMemcpyDeviceToHost));

    // generate reference for the given mean
    reference_calculation(vals, h_refHisto, numBins, numElems);

    // Now do the comparison
    checkResultsExact(h_refHisto, h_studentHisto, numBins);

    /////// Use Thrust
    GpuTimer thrustTimer;
    thrust::device_vector<unsigned int> tValsVec(numElems);
    thrust::copy(d_vals, d_vals + numElems, tValsVec.begin());

    thrustTimer.Start();
    thrustHistogram(tValsVec, d_histo, numBins, numElems);
    thrustTimer.Stop();
    printf("Thrust code ran in: %f msecs.\n", thrustTimer.Elapsed());

    // copy the thrust-computed histogram back to the host
    checkCudaErrors(cudaMemcpy(
        h_studentHisto, d_histo, sizeof(unsigned int) * numBins, cudaMemcpyDeviceToHost));
    // Now do the comparison
    checkResultsExact(h_refHisto, h_studentHisto, numBins);

    delete[] h_vals;
    delete[] h_refHisto;
    delete[] h_studentHisto;

    cudaFree(d_vals);
    cudaFree(d_histo);

    return 0;
}
