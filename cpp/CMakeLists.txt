cmake_minimum_required(VERSION 3.20)
project(MiniLLM CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Threads REQUIRED)
find_package(OpenMP REQUIRED)  # tools only; the engine uses its own thread pool (src/threads.h)

add_library(minillm src/gpt2.cpp src/kernels.cpp src/quant.cpp src/threads.cpp)
target_include_directories(minillm PUBLIC src)
target_link_libraries(minillm PUBLIC Threads::Threads)
# No -ffast-math: results must stay comparable with PyTorch's IEEE FP32.
target_compile_options(minillm PUBLIC -march=native -Wall -Wextra)
if(MINGW)
  target_link_options(minillm PUBLIC -static)  # standalone .exe, no MSYS2 DLLs needed on PATH
endif()

foreach(tool phase1_check phase2_bench phase3_check phase3_bench bandwidth)
  add_executable(${tool} tools/${tool}.cpp)
  target_link_libraries(${tool} PRIVATE minillm OpenMP::OpenMP_CXX)
  target_compile_definitions(${tool} PRIVATE MINILLM_ROOT="${CMAKE_SOURCE_DIR}/..")
endforeach()
