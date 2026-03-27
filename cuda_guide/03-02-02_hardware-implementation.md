---
title: "3.2.2. Hardware Implementation"
section: "3.2.2"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html#hardware-implementation"
---

## [3.2.2. Hardware Implementation](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced#hardware-implementation)[](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/#hardware-implementation "Permalink to this headline")

A streaming multiprocessor or SM (see [GPU Hardware Model](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html#programming-model-hardware-model)) is designed to execute hundreds of threads concurrently. To manage such a large number of threads, it employs a unique parallel computing model called _Single-Instruction, Multiple-Thread_, or _SIMT_, that is described in [SIMT Execution Model](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/#advanced-kernels-hardware-implementation-simt-architecture). The instructions are pipelined, leveraging instruction-level parallelism within a single thread, as well as extensive thread-level parallelism through simultaneous hardware multithreading as detailed in [Hardware Multithreading](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/#advanced-kernels-hardware-implementation-hardware-multithreading). Unlike CPU cores, SMs issue instructions in order and do not perform branch prediction or speculative execution.

Sections [SIMT Execution Model](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/#advanced-kernels-hardware-implementation-simt-architecture) and [Hardware Multithreading](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/#advanced-kernels-hardware-implementation-hardware-multithreading) describe the architectural features of the SM that are common to all devices. Section [Compute Capabilities](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html#compute-capabilities) provides the specifics for devices of different compute capabilities.

The NVIDIA GPU architecture uses a little-endian representation.
