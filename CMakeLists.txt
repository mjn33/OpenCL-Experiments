cmake_minimum_required(VERSION 2.8.9)

project(opencl_experiments)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if (NOT "${CMAKE_C_COMPILER_ID}" STREQUAL "MSVC")
  # Assume GCC/Clang, C99 support
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -std=c99")
  if ("${CMAKE_C_COMPILER_ID}" STREQUAL "GNU")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wdeclaration-after-statement")
  endif()
endif()

set(SOURCES
  main.c
)

add_library(opencl_experiments SHARED ${SOURCES})
