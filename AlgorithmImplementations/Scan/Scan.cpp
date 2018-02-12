#include <cuda.h>
#include <cuda_runtime.h>
#include <array>
#include <chrono>
#include <iostream>
#include <numeric>
#include <sstream>

extern void launchScanKernel(int* h_dataOut,
                             const int* h_dataIn,
                             const size_t dataSize,
                             float* execTime = nullptr);

extern void launchBlellochKernel(int* h_dataOut,
                                 const int* h_dataIn,
                                 const size_t dataSize,
                                 float* execTime = nullptr);

template <typename T, std::size_t size>
void serialExclusiveScan(const std::array<T, size>& dataIn, std::array<T, size>& dataOut) {
    dataOut[0] = 0;
    for (auto i = 1; i < dataIn.size(); i++) {
        dataOut[i] = dataOut[i - 1] + dataIn[i - 1];
    }
}

template <typename T, std::size_t size>
std::string printArray(const std::array<T, size>& inputArray) {
    std::stringstream out;
    for (const auto& val : inputArray) {
        out << val << " ";
    }
    return out.str();
}

int main(int argc, char** argv) {
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

    // Create input array
    static constexpr size_t ARRAY_SIZE = 64;
    std::array<int, ARRAY_SIZE> dataIn;
    dataIn.fill(1);
    // std::iota(dataIn.begin(), dataIn.end(), 1);
    std::array<int, ARRAY_SIZE> dataOut;

    // Run CPU implementation of exclusive scan
    auto start = std::chrono::high_resolution_clock::now();
    serialExclusiveScan(dataIn, dataOut);
    auto end = std::chrono::high_resolution_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    std::cout << "CPU execution took " << time.count() << " ns" << std::endl;

    std::cout << "CPU result = [ " << printArray(dataOut) << "]" << std::endl;

    // Run GPU implementation of exclusive scan
    std::array<int, ARRAY_SIZE> gpuDataOut;
    float gpuExecTime = 0;
    static constexpr int NS_IN_A_MS = 1000000; // 1000000 nanoseconds in one millisecond
    /*launchScanKernel(gpuDataOut.data(), dataIn.data(), ARRAY_SIZE, &gpuExecTime);

    std::cout << "***** Hillis-Steele Scan *****" << std::endl;
    std::cout << "GPU execution took " << gpuExecTime * NS_IN_A_MS << " ns" << std::endl;
    std::cout << "GPU result = [ " << printArray(gpuDataOut) << "]" << std::endl;
    */

    // Blelloch Scan
    launchBlellochKernel(gpuDataOut.data(), dataIn.data(), ARRAY_SIZE, &gpuExecTime);
    std::cout << "***** Blelloch Scan *****" << std::endl;
    std::cout << "GPU execution took " << gpuExecTime * NS_IN_A_MS << " ns" << std::endl;
    std::cout << "GPU result = [ " << printArray(gpuDataOut) << "]" << std::endl;

    return 0;
}