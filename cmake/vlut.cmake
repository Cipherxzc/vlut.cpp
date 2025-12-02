# This file contains vec-lut-specific compile definitions.

# Set the default options
option(VLUT_AVX512 "Enable AVX512 intrinsics" OFF)
option(VLUT_SVE "Enable SVE intrinsics" OFF)
option(VLUT_ACCELERATE "Enable Accelerate framework on Apple devices" OFF)

set(TABLE_ENTRY_SIZE 32 CACHE STRING "Tile size of the table entry")
set(WEIGHT_UNROLL_BLOCK 16 CACHE STRING "Weight unroll block size")

# Add compile definitions based on options
add_compile_definitions(TABLE_ENTRY_SIZE=${TABLE_ENTRY_SIZE})
message(STATUS "Adding definition: TABLE_ENTRY_SIZE=${TABLE_ENTRY_SIZE}")

add_compile_definitions(WEIGHT_UNROLL_BLOCK=${WEIGHT_UNROLL_BLOCK})
message(STATUS "Adding definition: WEIGHT_UNROLL_BLOCK=${WEIGHT_UNROLL_BLOCK}")

if(VLUT_AVX512)
    add_compile_definitions(VLUT_AVX512)
    message(STATUS "Adding definition: VLUT_AVX512")
endif()

if(VLUT_SVE)
    add_compile_definitions(VLUT_SVE)
    message(STATUS "Adding definition: VLUT_SVE")
endif()

if(VLUT_ACCELERATE)
    add_compile_definitions(VLUT_ACCELERATE)
    message(STATUS "Adding definition: VLUT_ACCELERATE")
endif()
