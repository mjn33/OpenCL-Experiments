cmake_minimum_required(VERSION 2.8.11)

project(opencl_experiments)

# Find OpenCL includes / libraries, useful macro can be found at:
# http://github.com/elhigu/cmake-findopencl
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
find_package( OpenCL REQUIRED )
message(STATUS "OpenCL includes: ${OPENCL_INCLUDE_DIRS}")
include_directories( ${OPENCL_INCLUDE_DIRS} )

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if ("${CMAKE_C_COMPILER_ID}" STREQUAL "MSVC")
  # Be explicit about using cdecl
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /Gd")
else()
  # Assume GCC/Clang, C99 support
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -std=c99")
  if ("${CMAKE_C_COMPILER_ID}" STREQUAL "GNU")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wdeclaration-after-statement")
  endif()
endif()

set(SOURCES
  plugin.c
)

add_library(opencl_experiments SHARED ${SOURCES})
set_target_properties(opencl_experiments PROPERTIES C_VISIBILITY_PRESET hidden)
target_include_directories(opencl_experiments PUBLIC ${CMAKE_BINARY_DIR})
target_link_libraries(opencl_experiments ${OPENCL_LIBRARIES})

include(GenerateExportHeader)
generate_export_header(opencl_experiments)
