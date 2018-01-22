#include <array>
#include <chrono>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <numeric>

extern void launchScanKernel(int *h_dataOut, const int *h_dataIn, const size_t dataSize);

template <typename T, std::size_t size>
void serialExclusiveScan(const std::array<T, size>& dataIn, std::array<T, size>& dataOut){
    dataOut[0] = 0;
    for (auto i = 1; i < dataIn.size(); i++) {
        dataOut[i] = dataOut[i-1] + dataIn[i - 1];
    }
}

int main(int argc, char **argv)
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        fprintf(stderr, "error: no devices supporting CUDA.\n");
        exit(EXIT_FAILURE);
    }
    int dev = 0;
    cudaSetDevice(dev);

    cudaDeviceProp devProps;
    if (cudaGetDeviceProperties(&devProps, dev) == 0)
    {
        printf("Using device %d:\n", dev);
        printf("%s; global mem: %dB; compute v%d.%d; clock: %d kHz\n",
               devProps.name, (int)devProps.totalGlobalMem, 
               (int)devProps.major, (int)devProps.minor, 
               (int)devProps.clockRate);
    }

    // Create input array
    static constexpr size_t ARRAY_SIZE = 8;
    std::array<int, ARRAY_SIZE> dataIn;
    std::iota(dataIn.begin(), dataIn.end(), 1);
    std::array<int, ARRAY_SIZE> dataOut;

    // Run CPU implementation of exclusive scan
    auto start = std::chrono::high_resolution_clock::now();
    serialExclusiveScan(dataIn, dataOut);
    auto end = std::chrono::high_resolution_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    std::cout << "CPU execution took " << time.count() << " ns" << std::endl; 

    std::cout << "CPU result = [ ";
    for (const auto& val : dataOut) {
        std::cout << val << " ";
    }
    std::cout << "]" << std::endl;

    // Run GPU implementation of exclusive scan
    std::array<int, ARRAY_SIZE> gpuDataOut;
    launchScanKernel(gpuDataOut.data(), dataIn.data(), ARRAY_SIZE);

    std::cout << "GPU result = [ ";
    for (const auto& val : gpuDataOut) {
        std::cout << val << " ";
    }
    std::cout << "]" << std::endl;

    return 0;
}