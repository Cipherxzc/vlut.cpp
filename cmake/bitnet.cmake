# This file contains bitnet-specific compile definitions. @xiangyu

# Set the default options
option(BITNET_DEBUG "Enable BITNET debug timing output" OFF)
option(BITNET_AVX2 "Enable BITNET AVX2 feature" OFF)
option(BITNET_LUT2 "Enable LUT2 instead of LUT1" ON)
option(BITNET_TILING "Enable tiling on prompt length" ON)
option(BITNET_PRINT_TENSORS "Enable printing tensors" OFF)

# Add compile definitions based on options
if(BITNET_DEBUG)
    add_compile_definitions(BITNET_DEBUG)
endif()
if(BITNET_AVX2)
    add_compile_definitions(BITNET_AVX2)
endif()
if(BITNET_LUT2)
    add_compile_definitions(BITNET_LUT2)
endif()
if(BITNET_TILING)
    add_compile_definitions(BITNET_TILING)
endif()
if(BITNET_PRINT_TENSORS)
    add_compile_definitions(BITNET_PRINT_TENSORS)
endif()