include (CTest)
enable_testing()

# Include gtest
include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})

##############
# Unit tests
##############

file(GLOB PS3_TEST_FILES ps3/*.cpp ps3/*.h)
CUDA_ADD_EXECUTABLE(ProblemSet3Tests ${PS3_TEST_FILES}
                    ../Problem\ Sets/Problem\ Set\ 3/student_func.cu)
target_link_libraries(ProblemSet3Tests gtest gtest_main pthread)

add_test(UnitTests ProblemSet3Tests)

set_target_properties(ProblemSet3Tests
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${EXECUTABLE_OUTPUT_PATH}/test"
)