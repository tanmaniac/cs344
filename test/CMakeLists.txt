include (CTest)
enable_testing()

# Include gtest
include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})

##############
# Unit tests
##############

file(GLOB PS3_TEST_FILES ps3/*.cpp)
add_executable(ProblemSet3Tests ${PS3_TEST_FILES})
target_link_libraries(ProblemSet3Tests gtest gtest_main pthread)

add_test(UnitTests ProblemSet3Tests)

set_target_properties(ProblemSet3Tests
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${EXECUTABLE_OUTPUT_PATH}/test"
)