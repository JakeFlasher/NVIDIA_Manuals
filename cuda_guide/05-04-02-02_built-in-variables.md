---
title: "5.4.2.2. Built-in Variables"
section: "5.4.2.2"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#built-in-variables"
---

### [5.4.2.2. Built-in Variables](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#built-in-variables)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#built-in-variables "Permalink to this headline")

The values used to specify and retrieve the kernel configuration for the grid and blocks along the x, y, and z dimensions are of type `dim3`. The variables used to obtain the block and thread indices are of type `uint3`. Both `dim3` and `uint3` are trivial structures consisting of three unsigned values named `x`, `y`, and `z`. In C++11 and later, the default value of all components of `dim3` is 1.

Built-in device-only variables:

- `dim3 gridDim`: contains the dimensions of the grid, namely the number of thread blocks, along the x, y, and z dimensions.
- `dim3 blockDim`: contains the dimensions of the thread block, namely the number of threads, along the x, y, and z dimensions.
- `uint3 blockIdx`: contains the block index within the grid, along the x, y, and z dimensions.
- `uint3 threadIdx`: contains the thread index within the block, along the x, y, and z dimensions.
- `int warpSize` :  A run-time value defined as the number of threads in a warp, commonly `32`. See also [Warps and SIMT](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html#programming-model-warps-simt) for the definition of a warp.
