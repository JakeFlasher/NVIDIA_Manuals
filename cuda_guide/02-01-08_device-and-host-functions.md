---
title: "2.1.8. Device and Host Functions"
section: "2.1.8"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/intro-to-cuda-cpp.html#device-and-host-functions"
---

## [2.1.8. Device and Host Functions](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics#device-and-host-functions)[](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/#device-and-host-functions "Permalink to this headline")

The `__global__` specifier is used to indicate the entry point for a kernel. That is, a function which will be invoked for parallel execution on the GPU. Most often, kernels are launched from the host, however it is possible to launch a kernel from within another kernel using [dynamic parallelism](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/dynamic-parallelism.html#cuda-dynamic-parallelism).

The specifier `__device__` indicates that a function should be compiled for the GPU and be callable from other `__device__` or `__global__` functions.  A function, including class member functions, functors, and lambdas, can be specified as both  `__device__` and `__host__` as in the example below.
