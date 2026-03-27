---
title: "4.18.4.2. C++ Language Interface for CDP"
section: "4.18.4.2"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/dynamic-parallelism.html#c-language-interface-for-cdp"
---

### [4.18.4.2. C++ Language Interface for CDP](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#c-language-interface-for-cdp)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#c-language-interface-for-cdp "Permalink to this headline")

The language interface and API available to CUDA kernels using CUDA C++ for Dynamic Parallelism is called the [CUDA Device Runtime](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/device-callable-apis.html#cuda-device-runtime).

Where possible the syntax and semantics of the CUDA Runtime API have been retained in order to facilitate ease of code reuse for routines that may run in either the host or device environments.

As with all code in CUDA C++, the APIs and code outlined here is per-thread code. This enables each thread to make unique, dynamic decisions regarding what kernel or operation to execute next. There are no synchronization requirements between threads within a block to execute any of the provided device runtime APIs, which enables the device runtime API functions to be called in arbitrarily divergent kernel code without deadlock.
