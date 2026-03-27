---
title: "3.2.1. Using PTX"
section: "3.2.1"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html#using-ptx"
---

## [3.2.1. Using PTX](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced#using-ptx)[](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/#using-ptx "Permalink to this headline")

_Parallel Thread Execution_ (PTX), the virtual machine instruction set architecture (ISA) that CUDA uses to abstract hardware ISAs, was introduced in [Section 1.3.3](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/cuda-platform.html#cuda-platform-ptx). Writing code in PTX directly is a highly advanced optimization technique that is not necessary for most developers and should be considered a tool of last resort. Nevertheless, there are situations where the fine-grained control enabled by writing PTX directly enables performance improvements in specific applications. These situations are typically in very performance-sensitive portions of an application where every fraction of a percent of performance improvement has significant benefits. All of the available PTX instructions are in the [PTX ISA document](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html).

`cuda::ptx` **namespace**

One way to use PTX directly in your code is to use the `cuda::ptx` namespace from [libcu++](https://nvidia.github.io/cccl/libcudacxx/). This namespace provides C++ functions that map directly to PTX instructions, simplifying their use within a C++ application. For more information, please refer to the [cuda::ptx namespace](https://nvidia.github.io/cccl/libcudacxx/ptx_api.html) documentation.

**Inline PTX**

Another way to include PTX in your code is to use inline PTX. This method is described in detail in the corresponding [documentation](https://docs.nvidia.com/cuda/inline-ptx-assembly/index.html). This is very similar to writing assembly code on a CPU.
