---
title: "3.5.1.2. Asynchronous Data Copies and the Tensor Memory Accelerator (TMA)"
section: "3.5.1.2"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/feature-survey.html#asynchronous-data-copies-and-the-tensor-memory-accelerator-tma"
---

### [3.5.1.2. Asynchronous Data Copies and the Tensor Memory Accelerator (TMA)](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced#asynchronous-data-copies-and-the-tensor-memory-accelerator-tma)[](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/#asynchronous-data-copies-and-the-tensor-memory-accelerator-tma "Permalink to this headline")

[Asynchronous data copies](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html#async-copies) in the context of CUDA kernel code refers to the ability to move data between shared memory and GPU DRAM while still carrying out computations. This should not be confused with asynchronous memory copies between the CPU and GPU. This feature makes used of asynchronous barriers. [Section 4.11](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html#async-copies) covers the use of asynchronous copies in detail.
