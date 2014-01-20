cmake_minimum_required(VERSION 2.8)
if (NOT CMAKE_BUILD_TYPE)
  message("-- No build type specified; defaulting to CMAKE_BUILD_TYPE=Release.")
  set(CMAKE_BUILD_TYPE Release CACHE STRING
    "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel."
    FORCE)
endif()
find_package(Boost REQUIRED)
include_directories(${Boost_INCLUDE_DIR})
add_executable(example example.cpp)
