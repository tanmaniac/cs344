#include <cuda.h>
#include <cuda_runtime.h>

#include <array>
#include <numeric>

namespace TestUtils {
int log2(int i) {
    int r = 0;
    while (i >>= 1)
        r++;
    return r;
}

int bit_reverse(int w, int bits) {
    int r = 0;
    for (int i = 0; i < bits; i++) {
        int bit = (w & (1 << i)) >> i;
        r |= bit << (bits - i - 1);
    }
    return r;
}

template <typename T, std::size_t Size>
struct MinMaxContainer {
public:
    // Host-side data
    std::array<T, Size> data;
    T min;
    T max;
    // CUDA device-side data
    T* d_data;

    MinMaxContainer() {
        std::iota(data.begin(), data.end(), 0);
        cudaMalloc((void**)&d_data, Size * sizeof(T));
        cudaMemcpy(d_data, data.data(), Size * sizeof(T), cudaMemcpyHostToDevice);
    }

    ~MinMaxContainer() {
        cudaFree(d_data);
    }

    void fetchDataFromDevice() {
        cudaMemcpy(data.data(), d_data, Size * sizeof(T), cudaMemcpyDeviceToHost);
    }
};

template <typename T, std::size_t dataSize, std::size_t numBins>
struct HistoContainer {
public:
    // Host-side data
    std::array<T, dataSize> _data;
    std::array<unsigned int, numBins> _bins;
    // Device-side data
    T* _d_data;
    unsigned int* _d_bins;

    HistoContainer() {
        for (unsigned int i = 0; i < dataSize; i++) {
            _data[i] = i % numBins;
        }
        cudaMalloc((void**)&_d_data, dataSize * sizeof(T));
        cudaMalloc((void**)&_d_bins, numBins * sizeof(unsigned int));
        cudaMemcpy(_d_data, _data.data(), dataSize * sizeof(T), cudaMemcpyHostToDevice);
    }

    ~HistoContainer() {
        cudaFree(_d_data);
        cudaFree(_d_bins);
    }

    void fetchDataFromDevice() {
        cudaMemcpy(_bins.data(), _d_bins, numBins * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    }
};

}; // namespace TestUtils