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

## optional flags
#add_definitions(-DYELL_USE_MEM_POOL)
#add_definitions(-DYELL_USE_32BITS_MODULI)
include_directories(include/)

file(GLOB YELL_ASMC lib/prng/*.s)
file(GLOB YELL_SRC lib/prng/*.cpp lib/*.cpp lib/prng/*.s)
add_library(yell STATIC ${YELL_SRC} ${YELL_ASMC})

add_executable(example example.cpp)
target_link_libraries(example yell ${GMP_LIBRARIES} ${MPFR_LIBRARIES})
add_executable(mont rescale.cc)
target_link_libraries(mont yell ${GMP_LIBRARIES} ${MPFR_LIBRARIES})

include_directories(${GMP_INCLUDES} ${MPFR_INCLUDES})

