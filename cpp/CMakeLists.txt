cmake_minimum_required(VERSION 3.20)
project(MiniLLM CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(OpenMP REQUIRED)

add_library(minillm src/gpt2.cpp src/kernels.cpp)
target_include_directories(minillm PUBLIC src)
target_link_libraries(minillm PUBLIC OpenMP::OpenMP_CXX)
# No -ffast-math: results must stay comparable with PyTorch's IEEE FP32.
target_compile_options(minillm PUBLIC -march=native -Wall -Wextra)
if(MINGW)
  target_link_options(minillm PUBLIC -static)  # standalone .exe, no MSYS2 DLLs needed on PATH
endif()

add_executable(phase1_check tools/phase1_check.cpp)
target_link_libraries(phase1_check PRIVATE minillm)
target_compile_definitions(phase1_check PRIVATE MINILLM_ROOT="${CMAKE_SOURCE_DIR}/..")
