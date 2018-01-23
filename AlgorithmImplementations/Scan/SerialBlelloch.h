/**
 * Serial implementation of Blelloch's prefix sum algorithm
 */

#include <array>
#include <iostream>

namespace scan {

/**
 * Up-sweep step of the Blelloch scan. Algorithm follows:
 *  for d = 0 to log_2(n - 1), do:
 *      for k = 0 to n - 1 by 2^(d+1), do:
 *          x[k + 2^(d+1) - 1] = x[k + 2^d - 1] + x[k + 2^d]
 *
 * https://developer.nvidia.com/gpugems/GPUGems3/gpugems3_ch39.html
 */
template <typename T, std::size_t S>
void reduce(const std::array<T, S>& dataIn, std::array<T, S>& dataOut) {
    static constexpr size_t ARRAY_SIZE = dataIn.size();
}

}; // namespace scan