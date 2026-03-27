---
title: "2.5. NVCC: The NVIDIA CUDA Compiler"
section: "2.5"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/nvcc.html#nvcc-the-nvidia-cuda-compiler"
---

# [2.5. NVCC: The NVIDIA CUDA Compiler](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics#nvcc-the-nvidia-cuda-compiler)[](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/#nvcc-the-nvidia-cuda-compiler "Permalink to this headline")

[The NVIDIA CUDA Compiler](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html) `nvcc` is a toolchain from NVIDIA for compiling CUDA C/C++ as well as [PTX](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html) code. The toolchain is part of the [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) and consists of several tools, including the compiler, linker, and the PTX and [Cubin](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/cuda-platform.html#cuda-platform-cubins-fatbins) assemblers. The top-level `nvcc` tool coordinates the compilation process, invoking the appropriate tool for each stage of compilation.

`nvcc` drives offline compilation of CUDA code, in contrast to online or Just-in-Time (JIT) compilation driven by the CUDA runtime compiler [nvrtc](https://docs.nvidia.com/cuda/nvrtc/index.html).

This chapter covers the most common uses and details of `nvcc` needed for building applications. Full coverage of `nvcc` is found in the [nvcc documentation](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html).
