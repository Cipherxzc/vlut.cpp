# This file contains bitnet-specific compile definitions. @xiangyu

# Set the default options
option(BITNET_DEBUG "Enable BITNET debug timing output" OFF)
option(BITNET_AVX2 "Enable BITNET AVX2 feature" OFF)
option(BITNET_LUT2 "Enable LUT2 instead of LUT1" OFF)
option(BITNET_TILING "Enable tiling on prompt length" ON)
option(BITNET_PRINT_TENSORS "Enable printing tensors" OFF)
option(BITNET_AVX512 "Enable AVX512 intrinsics" OFF)
option(BITNET_SVE "Enable SVE intrinsics" OFF)
option(BITNET_ACCELERATE "Enable Accelerate framework on Apple devices" OFF)

set(TABLE_ENTRY_SIZE 32 CACHE STRING "Tile size of the table entry")

# Add compile definitions based on options
if(BITNET_DEBUG)
    add_compile_definitions(BITNET_DEBUG)
endif()

if(BITNET_AVX2)
    add_compile_definitions(BITNET_AVX2)
endif()

if(BITNET_LUT2)
    add_compile_definitions(BITNET_LUT2)
else()
    add_compile_definitions(BITNET_LUT)
endif()

if(BITNET_TILING)
    add_compile_definitions(BITNET_TILING)
    add_compile_definitions(TABLE_ENTRY_SIZE=${TABLE_ENTRY_SIZE})
endif()

if(BITNET_PRINT_TENSORS)
    add_compile_definitions(BITNET_PRINT_TENSORS)
endif()

if(BITNET_AVX512)
    add_compile_definitions(BITNET_AVX512)
endif()

if(BITNET_SVE)
    add_compile_definitions(BITNET_SVE)
endif()

if(BITNET_ACCELERATE)
    add_compile_definitions(BITNET_ACCELERATE)
endif()