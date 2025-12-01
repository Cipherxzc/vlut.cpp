# This file contains vec-lut-specific compile definitions.

# Set the default options
option(VLUT_AVX512 "Enable AVX512 intrinsics" OFF)
option(VLUT_SVE "Enable SVE intrinsics" OFF)
option(VLUT_ACCELERATE "Enable Accelerate framework on Apple devices" OFF)

set(TABLE_ENTRY_SIZE 32 CACHE STRING "Tile size of the table entry")
set(WEIGHT_UNROLL_BLOCK 16 CACHE STRING "Weight unroll block size")

# Auto-detect SVE support
include(CheckCSourceCompiles)
include(CheckCSourceRuns)

# Save original flags to restore later
set(CMAKE_REQUIRED_FLAGS_ORIG ${CMAKE_REQUIRED_FLAGS})

# First check if the compiler supports SVE
set(CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS_ORIG} -march=armv8.2-a+sve")
check_c_source_compiles("
#include <arm_sve.h>
int main() {
    svbool_t pg = svptrue_b8();
    return 0;
}
" COMPILER_SUPPORTS_SVE)

# Then create a test program that checks for runtime support
if(COMPILER_SUPPORTS_SVE)
  set(CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS_ORIG} -march=armv8.2-a+sve")
  check_c_source_runs("
  #include <arm_sve.h>
  #include <stdio.h>
  #include <signal.h>
  #include <setjmp.h>

  static jmp_buf env;

  static void sigill_handler(int sig) {
    longjmp(env, 1);
  }

  int main() {
    // Set up signal handler for SIGILL
    signal(SIGILL, sigill_handler);
    
    // If setjmp returns 0, we're setting up the jump point
    // If it returns non-zero, we've returned from a SIGILL
    if (setjmp(env) == 0) {
      // Try to execute an SVE instruction
      svbool_t pg = svptrue_b8();
      // If we reach here, the instruction executed successfully
      return 0;
    } else {
      // We got a SIGILL, which means SVE is not supported
      return 1;
    }
  }
  " HAVE_SVE_HARDWARE)
endif()

# Resume original flags
set(CMAKE_REQUIRED_FLAGS ${CMAKE_REQUIRED_FLAGS_ORIG})

# Set the final HAVE_SVE variable
if(COMPILER_SUPPORTS_SVE AND HAVE_SVE_HARDWARE)
  set(HAVE_SVE TRUE)
else()
  set(HAVE_SVE FALSE)
endif()

# TODO: This detect is buggy on some devices (e.g., NEON smartphone)
# Currently, mannually comment out the `set(VLUT_SVE ON)` line
if(HAVE_SVE)
    message(STATUS "SVE support detected, enabling VLUT_SVE")
    set(VLUT_SVE ON)
endif()

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