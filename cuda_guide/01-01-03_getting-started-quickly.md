---
title: "1.1.3. Getting Started Quickly"
section: "1.1.3"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/introduction.html#getting-started-quickly"
---

## [1.1.3. Getting Started Quickly](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction#getting-started-quickly)[](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/#getting-started-quickly "Permalink to this headline")

There are many ways to leverage the compute power provided by GPUs. This guide covers programming for the CUDA GPU platform in high-level languages such as C++. However, there are many ways to utilize GPUs in applications that do not require directly writing GPU code.

An ever-growing collection of algorithms and routines from a variety of domains is available through specialized libraries. When a library has already been implemented—especially those provided by NVIDIA—using it is often more productive and performant than reimplementing algorithms from scratch. Libraries like cuBLAS, cuFFT, cuDNN, and CUTLASS are just a few examples of libraries that help developers avoid reimplementing well-established algorithms. These libraries have the added benefit of being optimized for each GPU architecture, providing an ideal mix of productivity, performance, and portability.

There are also frameworks, particularly those used for artificial intelligence, that provide GPU-accelerated building blocks. Many of these frameworks achieve their acceleration by leveraging the GPU-accelerated libraries mentioned above.

Additionally, domain-specific languages (DSLs) such as NVIDIA’s Warp or OpenAI’s Triton compile to run directly on the CUDA platform. This provides an even higher-level method of programming GPUs than the high-level languages covered in this guide.

The [NVIDIA Accelerated Computing Hub](https://github.com/NVIDIA/accelerated-computing-hub) contains resources, examples, and tutorials to teach GPU and CUDA computing.
