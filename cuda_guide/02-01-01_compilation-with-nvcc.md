---
title: "2.1.1. Compilation with NVCC"
section: "2.1.1"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/intro-to-cuda-cpp.html#compilation-with-nvcc"
---

## [2.1.1. Compilation with NVCC](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics#compilation-with-nvcc)[](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/#compilation-with-nvcc "Permalink to this headline")

GPU code written in C++ is compiled using the NVIDIA Cuda Compiler, `nvcc`. `nvcc` is a compiler driver that simplifies the process of compiling C++ or PTX code: It provides simple and familiar command line options and executes them by invoking the collection of tools that implement the different compilation stages.

This guide will show `nvcc` command lines which can be used on any Linux system with the CUDA Toolkit installed, at a Windows command line or power shell, or on Windows Subsystem for Linux with the CUDA Toolkit.  The [nvcc chapter](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/nvcc.html#nvcc) of this guide covers common use cases of `nvcc`, and complete documentation is provided by [the nvcc user manual](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html).
