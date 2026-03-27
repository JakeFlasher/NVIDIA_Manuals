---
title: "1.3.4.1. Binary Compatibility"
section: "1.3.4.1"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/cuda-platform.html#binary-compatibility"
---

### [1.3.4.1. Binary Compatibility](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction#binary-compatibility)[](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/#binary-compatibility "Permalink to this headline")

NVIDIA GPUs guarantee binary compatibility in certain circumstances. Specifically, within a major version of compute capability, GPUs with minor compute capability greater than or equal to the targeted version of cubin can load and execute that cubin. For example, if an application contains a cubin with code compiled for compute capability 8.6, that cubin can be loaded and executed on GPUs with compute capability 8.6 or 8.9. It cannot, however, be loaded on GPUs with compute capability 8.0, because the GPU’s CC minor version, 0, is lower than the code’s minor version, 6.

NVIDIA GPUs are not binary compatible between major compute capability versions. That is, cubin code compiled for compute capability 8.6 will not load on GPUs of compute capability 9.0.

When discussing binary code, the binary code is often referred to as having a version such as *sm_86* in the above example. This is the same as saying the binary was built for compute capability 8.6. This shorthand is often used because it is how a developer specifies this binary build target to the NVIDIA CUDA compiler, [nvcc](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/nvcc.html#nvcc).

> **Note**
>
> Binary compatibility is promised only for binaries created by NVIDIA tools such as `nvcc`. Manual editing or generating binary code for NVIDIA GPUs is not supported. Compatibility promises are invalidated if binaries are modified in any way.
