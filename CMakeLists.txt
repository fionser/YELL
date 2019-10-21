cmake_minimum_required(VERSION 2.8)
project(YELL CXX ASM-ATT)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

include(CMakeCompilers.txt)
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

find_package(GMP REQUIRED)
find_package(MPFR REQUIRED)

include_directories(${GMP_INCLUDES} ${MPFR_INCLUDES})

set(RUNLIBS yell ${GMP_LIBRARIES} ${MPFR_LIBRARIES})

## optional flags
#add_definitions(-DYELL_USE_MEM_POOL)
#add_definitions(-DYELL_USE_32BITS_MODULI)
include_directories(include/)

file(GLOB YELL_ASMC lib/prng/*.s)
file(GLOB YELL_SRC lib/prng/*.cpp lib/*.cpp lib/prng/*.s)
add_library(yell STATIC ${YELL_SRC} ${YELL_ASMC})

add_executable(example example.cpp)
target_link_libraries(example ${RUNLIBS})

find_package(benchmark)
include_directories(/Users/juhou/include)
link_directories(/Users/juhou/lib)
if(benchmark_FOUND)
  link_directories(/usr/local/lib)
  add_executable(benchmark_ntt benchmark_ntt.cc)
  target_link_libraries(benchmark_ntt benchmark nfllib ${RUNLIBS})
endif()

find_package(GTest)

if (GTEST_FOUND)

macro(package_add_test TESTNAME)
    add_executable(${TESTNAME} ${ARGN})
    target_link_libraries(${TESTNAME} ${GTEST_LIBRARY} ${GTEST_MAIN_LIBRARIES} ${RUNLIBS})
    add_test(NAME ${TESTNAME} COMMAND ${TESTNAME})
endmacro()

include_directories(${GTEST_INCLUDE_DIRS})
enable_testing()

#package_add_test(test_basic test/test_basic.cpp)

endif()

