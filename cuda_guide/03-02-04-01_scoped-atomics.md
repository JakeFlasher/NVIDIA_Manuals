---
title: "3.2.4.1. Scoped Atomics"
section: "3.2.4.1"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html#scoped-atomics"
---

### [3.2.4.1. Scoped Atomics](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced#scoped-atomics)[](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/#scoped-atomics "Permalink to this headline")

[Section 5.4.5](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#atomic-functions) gives an overview of atomic functions available in CUDA. In this section, we will focus on _scoped_ atomics that support [C++ standard atomic memory](https://en.cppreference.com/w/cpp/atomic/memory_order.html) semantics, available through the [libcu++](https://nvidia.github.io/cccl/libcudacxx/extended_api/synchronization_primitives.html) library or through compiler built-in functions. Scoped atomics provide the tools for efficient synchronization at the appropriate level of the CUDA thread hierarchy, enabling both correctness and performance in complex parallel algorithms.
