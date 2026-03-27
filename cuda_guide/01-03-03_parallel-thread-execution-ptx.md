---
title: "1.3.3. Parallel Thread Execution (PTX)"
section: "1.3.3"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/cuda-platform.html#parallel-thread-execution-ptx"
---

## [1.3.3. Parallel Thread Execution (PTX)](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction#parallel-thread-execution-ptx)[](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/#parallel-thread-execution-ptx "Permalink to this headline")

A fundamental but sometimes invisible layer of the CUDA platform is the _Parallel Thread Execution_ (PTX) virtual instruction set architecture (ISA). PTX is a high-level assembly language for NVIDIA GPUs. PTX provides an abstraction layer over the physical ISA of real GPU hardware. Like other platforms, applications can be written directly in this assembly language, though doing so can add unnecessary complexity and difficulty to software development.

Domain-specific languages and compilers for high-level languages can generate PTX code as an intermediate representation (IR) and then use NVIDIA’s offline or just-in-time (JIT) compilation tools to produce executable binary GPU code. This enables the CUDA platform to be programmable from languages other than just those supported by NVIDIA-provided tools such as [NVCC: The NVIDIA CUDA Compiler](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/nvcc.html#nvcc).

Since GPU capabilities change and grow over time, the PTX virtual ISA specification is versioned. PTX versions, like SM versions, correspond to a compute capability. For example, PTX which supports all the features of compute capability 8.0 is called *compute_80*.

Full documentation on PTX can be found in the [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html) .
