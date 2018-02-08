############################################################################
# <summary> CMakeLists.txt for OpenCV and CUDA. </summary>
# <date>    2012-02-07          </date>
# <author>  Quan Tran Minh. edit by Johannes Kast, Michael Sarahan </author>
# <email>   quantm@unist.ac.kr  kast.jo@googlemail.com msarahan@gmail.com</email>
############################################################################

# collect source files
file( GLOB  hdr *.hpp *.h)
file( GLOB  cu  *.cu)
SET (HW4_files main.cpp loadSaveImage.cpp reference_calc.cpp compare.cpp)

# ptxas errors without this (https://stackoverflow.com/a/29604081)
# SET(CUDA_SEPARABLE_COMPILATION ON)
# HW4 executable crashes with "-G" debug flag turned on
SET (CUDA_NVCC_FLAGS_DEBUG "")
CUDA_ADD_EXECUTABLE(HW4 ${HW4_files} ${hdr} ${img} ${cu})


