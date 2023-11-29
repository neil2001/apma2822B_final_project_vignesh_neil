cmake_minimum_required(VERSION 3.10)

project(GPU_Raytracer CUDA)

set(CMAKE_CUDA_STANDARD 14)

include_directories(include)

add_executable(a.out src/main.cu src/kernel.cu src/utils.cu)
