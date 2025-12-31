cmake_minimum_required(VERSION 3.18)
project(ed25519brute_cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_CUDA_ARCHITECTURES 89)

# Optimization flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math")
set(CMAKE_CXX_FLAGS_RELEASE "/O2 /DNDEBUG")

add_executable(ed25519brute_cuda
    src/main.cu
)

target_include_directories(ed25519brute_cuda PRIVATE src)
