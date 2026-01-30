cmake_minimum_required(VERSION 3.12)

project(cuda1 CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(cuda1 tmp2.cu
        kernels.h
        matr_methods.h
        device_funcs.h)



set_target_properties(cuda1 PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
