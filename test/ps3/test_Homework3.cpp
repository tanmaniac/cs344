#include "test_Homework3.h"

#include "gtest/gtest.h"

// Functions from student_func.cu

void launchMinMaxKernel(const float* d_logLuminance,
                        float& min_logLum,
                        float& max_logLum,
                        const size_t numRows,
                        const size_t numCols);

void launchHistoKernel(const float* const d_logLuminance,
                       float min_logLum,
                       float lumRange,
                       const size_t dataSize,
                       const size_t numBins,
                       unsigned int* const d_bins);

TEST(MinMaxKernel, ZeroAndOne) {
    static constexpr std::size_t numRows = 1;
    static constexpr std::size_t numCols = 2;

    TestUtils::MinMaxContainer<float, numRows * numCols> testData;

    launchMinMaxKernel(testData.d_data, testData.min, testData.max, numRows, numCols);

    ASSERT_EQ(testData.min, 0.0);
    ASSERT_EQ(testData.max, 1.0);
}

TEST(MinMaxKernel, FullBlock) {
    static constexpr std::size_t numRows = 1;
    static constexpr std::size_t numCols = 1024;

    TestUtils::MinMaxContainer<float, numRows * numCols> testData;

    launchMinMaxKernel(testData.d_data, testData.min, testData.max, numRows, numCols);

    ASSERT_EQ(testData.min, 0.0);
    ASSERT_EQ(testData.max, numCols - 1.0);
}

TEST(MinMaxKernel, MultipleFullBlocks) {
    static constexpr std::size_t numRows = 3;
    static constexpr std::size_t numCols = 1024;

    TestUtils::MinMaxContainer<float, numRows * numCols> testData;

    launchMinMaxKernel(testData.d_data, testData.min, testData.max, numRows, numCols);

    ASSERT_EQ(testData.min, 0.0);
    ASSERT_EQ(testData.max, numRows * numCols - 1.0);
}

TEST(MinMaxKernel, NonFullBlocks) {
    static constexpr std::size_t numRows = 7;
    static constexpr std::size_t numCols = 843;

    TestUtils::MinMaxContainer<float, numRows * numCols> testData;

    launchMinMaxKernel(testData.d_data, testData.min, testData.max, numRows, numCols);

    ASSERT_EQ(testData.min, 0.0);
    ASSERT_EQ(testData.max, numRows * numCols - 1.0);
}

TEST(HistoKernel, BinSize10) {
    static constexpr size_t binSize = 10;
    static constexpr size_t datasetSize = 100;

    TestUtils::HistoContainer<float, datasetSize, binSize> testData;

    launchHistoKernel(testData._d_data, 0, 10, datasetSize, binSize, testData._d_bins);
    testData.fetchDataFromDevice();
    for (const auto& val : testData._bins) {
        ASSERT_EQ(val, binSize);
    }
}
