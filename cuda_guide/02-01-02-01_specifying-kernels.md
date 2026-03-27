---
title: "2.1.2.1. Specifying Kernels"
section: "2.1.2.1"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/intro-to-cuda-cpp.html#specifying-kernels"
---

### [2.1.2.1. Specifying Kernels](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics#specifying-kernels)[](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/#specifying-kernels "Permalink to this headline")

The code for a kernel is specified using the `__global__` declaration specifier. This indicates to the compiler that this function will be compiled for the GPU in a way that allows it to be invoked from a kernel launch. A kernel launch is an operation which starts a kernel running, usually from the CPU. Kernels are functions with a `void` return type.

```cuda
// Kernel definition
__global__ void vecAdd(float* A, float* B, float* C)
{

}
```
