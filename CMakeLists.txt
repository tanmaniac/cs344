############################################################################
# <summary> CMakeLists.txt for OpenCV and CUDA. </summary>
# <date>    2012-02-07          </date>
# <author>  Quan Tran Minh. edit by Johannes Kast, Michael Sarahan </author>
# <email>   quantm@unist.ac.kr  kast.jo@googlemail.com msarahan@gmail.com</email>
############################################################################

cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
project(cs344)

# Only use required modules from OpenCV. This is different for OpenCV 3.x vs 2.x
find_package(OpenCV REQUIRED core imgproc highgui)
find_package(CUDA REQUIRED)

link_libraries(${OpenCV_LIBS} )

set (EXECUTABLE_OUTPUT_PATH "${CMAKE_SOURCE_DIR}/bin/")

if(CUDA_FOUND)
  # compared to class settings, we let NVidia's FindCUDA CMake detect 
  # whether to build x64.  We tell it to support most devices, though, 
  # to make sure more people can easily run class code without knowing 
  # about this compiler argument
  set(CUDA_NVCC_FLAGS "
  -ccbin g++;
  -gencode;arch=compute_50,code=sm_50;
  -gencode;arch=compute_52,code=sm_52;
  -gencode;arch=compute_60,code=sm_60;
  -gencode;arch=compute_61,code=sm_61;
  -gencode;arch=compute_62,code=sm_62")

  set(CMAKE_CXX_FLAGS_DEBUG "-DSAVE_DEBUG_IMGS -DDRAW_DEBUG_IMGS")

  # add -Wextra compiler flag for gcc compilations
  if (UNIX)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Xcompiler -Wextra")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-arcs -ftest-coverage")
  endif (UNIX)

  # Support g++ versions less than 6.0
  if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 6.0)
      set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
      set (CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11")
      set (CUDA_PROPAGATE_HOST_FLAGS OFF)
    endif()
  endif()

  # add debugging to CUDA NVCC flags.  For NVidia's NSight tools.
  set(CUDA_NVCC_FLAGS_DEBUG ${CUDA_NVCC_FLAGS_DEBUG} "-G")

  add_subdirectory (Problem\ Sets/Problem\ Set\ 1)
  add_subdirectory (Problem\ Sets/Problem\ Set\ 2)
  add_subdirectory (Problem\ Sets/Problem\ Set\ 3)
  add_subdirectory (Problem\ Sets/Problem\ Set\ 4)
  add_subdirectory (Problem\ Sets/Problem\ Set\ 5)
  add_subdirectory (Problem\ Sets/Problem\ Set\ 6)

  add_subdirectory (AlgorithmImplementations)
  add_subdirectory(external)
  add_subdirectory(test)

else(CUDA_FOUND)
  message("CUDA is not installed on this system.")
endif()
