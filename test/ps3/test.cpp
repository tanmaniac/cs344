#include <cuda.h>
#include <cuda_runtime.h>
#include "gtest/gtest.h"

#include <array>
#include <numeric>

// Functions from student_func.cu

void launchMinMaxKernel(const float* d_logLuminance,
                        float& min_logLum,
                        float& max_logLum,
                        const size_t numRows,
                        const size_t numCols);

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

TEST(MinMaxKernel, ZeroAndOne) {
    static constexpr std::size_t numRows = 1;
    static constexpr std::size_t numCols = 2;

    MinMaxContainer<float, numRows * numCols> testData;
    
    launchMinMaxKernel(testData.d_data, testData.min, testData.max, numRows, numCols);

    ASSERT_EQ(testData.min, 0.0);
    ASSERT_EQ(testData.max, 1.0);
}

TEST(MinMaxKernel, FullBlock) {
    static constexpr std::size_t numRows = 1;
    static constexpr std::size_t numCols = 1024;

    MinMaxContainer<float, numRows * numCols> testData;
    
    launchMinMaxKernel(testData.d_data, testData.min, testData.max, numRows, numCols);

    ASSERT_EQ(testData.min, 0.0);
    ASSERT_EQ(testData.max, numCols - 1.0);
}

TEST(MinMaxKernel, MultipleFullBlocks) {
    static constexpr std::size_t numRows = 3;
    static constexpr std::size_t numCols = 1024;

    MinMaxContainer<float, numRows * numCols> testData;
    
    launchMinMaxKernel(testData.d_data, testData.min, testData.max, numRows, numCols);

    ASSERT_EQ(testData.min, 0.0);
    ASSERT_EQ(testData.max, numRows * numCols - 1.0);
}

TEST(MinMaxKernel, NonFullBlocks) {
    static constexpr std::size_t numRows = 7;
    static constexpr std::size_t numCols = 843;

    MinMaxContainer<float, numRows * numCols> testData;
    
    launchMinMaxKernel(testData.d_data, testData.min, testData.max, numRows, numCols);

    ASSERT_EQ(testData.min, 0.0);
    ASSERT_EQ(testData.max, numRows * numCols - 1.0);
}
