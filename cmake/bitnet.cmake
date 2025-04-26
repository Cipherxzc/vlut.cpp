# This file contains bitnet-specific compile definitions. @xiangyu

# Set the default options
option(BITNET_DEBUG "Enable BITNET debug timing output" OFF)
option(BITNET_AVX2 "Enable BITNET AVX2 feature" OFF)
option(BITNET_LUT2 "Enable LUT2 instead of LUT1" ON)
option(BITNET_TILING "Enable tiling on prompt length" ON)
option(BITNET_PRINT_TENSORS "Enable printing tensors" OFF)
option(BITNET_AVX512 "Enable AVX512 intrinsics" OFF)
option(BITNET_SVE "Enable SVE intrinsics" OFF)
option(BITNET_ACCELERATE "Enable Accelerate framework on Apple devices" OFF)

set(TABLE_ENTRY_SIZE 32 CACHE STRING "Tile size of the table entry")
set(WEIGHT_UNROLL_BLOCK 16 CACHE STRING "Weight unroll block size")

# Auto-detect SVE support
include(CheckCSourceCompiles)

# Save original flags to restore later
set(CMAKE_REQUIRED_FLAGS_ORIG ${CMAKE_REQUIRED_FLAGS})

# Temporarily set SVE architecture flags ONLY for detection
set(CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS_ORIG} -march=armv8.2-a+sve")
check_c_source_compiles("
#include <arm_sve.h>
int main() {
    svbool_t pg = svptrue_b8();
    return 0;
}
" HAVE_SVE)

# Immediately restore original flags to avoid affecting other code
set(CMAKE_REQUIRED_FLAGS ${CMAKE_REQUIRED_FLAGS_ORIG})

if(HAVE_SVE)
    message(STATUS "SVE support detected, enabling BITNET_SVE")
    set(BITNET_SVE ON)
endif()

# Add compile definitions based on options
if(BITNET_DEBUG)
    add_compile_definitions(BITNET_DEBUG)
    message(STATUS "Adding definition: BITNET_DEBUG")
endif()

if(BITNET_AVX2)
    add_compile_definitions(BITNET_AVX2)
    message(STATUS "Adding definition: BITNET_AVX2")
endif()

if(BITNET_LUT2)
    add_compile_definitions(BITNET_LUT2)
    message(STATUS "Adding definition: BITNET_LUT2")
else()
    add_compile_definitions(BITNET_LUT)
    message(STATUS "Adding definition: BITNET_LUT")
endif()

if(BITNET_TILING)
    add_compile_definitions(BITNET_TILING)
    message(STATUS "Adding definition: BITNET_TILING")
    
    add_compile_definitions(TABLE_ENTRY_SIZE=${TABLE_ENTRY_SIZE})
    message(STATUS "Adding definition: TABLE_ENTRY_SIZE=${TABLE_ENTRY_SIZE}")
endif()

add_compile_definitions(WEIGHT_UNROLL_BLOCK=${WEIGHT_UNROLL_BLOCK})
message(STATUS "Adding definition: WEIGHT_UNROLL_BLOCK=${WEIGHT_UNROLL_BLOCK}")

if(BITNET_PRINT_TENSORS)
    add_compile_definitions(BITNET_PRINT_TENSORS)
    message(STATUS "Adding definition: BITNET_PRINT_TENSORS")
endif()

if(BITNET_AVX512)
    add_compile_definitions(BITNET_AVX512)
    message(STATUS "Adding definition: BITNET_AVX512")
endif()

if(BITNET_SVE)
    add_compile_definitions(BITNET_SVE)
    message(STATUS "Adding definition: BITNET_SVE")
endif()

if(BITNET_ACCELERATE)
    add_compile_definitions(BITNET_ACCELERATE)
    message(STATUS "Adding definition: BITNET_ACCELERATE")
endif()