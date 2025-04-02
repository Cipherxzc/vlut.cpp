# This file contains bitnet-specific compile definitions. @xiangyu

# Set the default options
option(BITNET_DEBUG "Enable BITNET debug timing output" OFF)
option(BITNET_AVX2 "Enable BITNET AVX2 feature" OFF)
option(BITNET_LUT "Enable LUT1" ON)
option(BITNET_LUT2 "Enable LUT2 instead of LUT1" OFF)
option(BITNET_TILING "Enable tiling on prompt length" ON)
option(BITNET_PRINT_TENSORS "Enable printing tensors" OFF)

set(TABLE_ENTRY_SIZE 32 CACHE STRING "Tile size of the table entry")

# Add compile definitions based on options
if(BITNET_DEBUG)
    add_compile_definitions(BITNET_DEBUG)
endif()

if(BITNET_AVX2)
    add_compile_definitions(BITNET_AVX2)
endif()

if(BITNET_LUT AND BITNET_LUT2)
    message(FATAL_ERROR "BITNET_LUT and BITNET_LUT2 cannot be enabled at the same time.")
elseif(BITNET_LUT)
    add_compile_definitions(BITNET_LUT)
elseif(BITNET_LUT2)
    add_compile_definitions(BITNET_LUT2)
endif()

if(BITNET_TILING)
    add_compile_definitions(BITNET_TILING)
    add_compile_definitions(TABLE_ENTRY_SIZE=${TABLE_ENTRY_SIZE})
endif()

if(BITNET_PRINT_TENSORS)
    add_compile_definitions(BITNET_PRINT_TENSORS)
endif()
