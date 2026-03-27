---
title: "1.2.2. GPU Hardware Model"
section: "1.2.2"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html#gpu-hardware-model"
---

## [1.2.2. GPU Hardware Model](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction#gpu-hardware-model)[](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/#gpu-hardware-model "Permalink to this headline")

Like any programming model, CUDA relies on a conceptual model of the underlying hardware. For the purposes of CUDA programming, the GPU can be considered to be a collection of _Streaming Multiprocessors_ (SMs) which are organized into groups called _Graphics Processing Clusters_ (GPCs). Each SM contains a local register file, a unified data cache, and a number of functional units that perform computations. The unified data cache provides the physical resources for _shared memory_ and L1 cache. The allocation of the unified data cache to L1 and shared memory can be configured at runtime. The sizes of different types of memory and the number of functional units within an SM can vary across GPU architectures.

> **Note**
>
> The actual hardware layout of a GPU or the way it physically carries out the execution of the programming model may vary. These differences do not affect correctness of software written using the CUDA programming model.

![The CUDA programming model view of CPU and GPU components and connection](images/___-____w___-______1.png)

Figure 2 A GPU has many streaming multiprocessors (SMs), each of which contains many functional units. Graphics processing clusters (GPCs) are collections of SMs. A GPU is a set of GPCs connected to the GPU memory. A CPU typically has several cores and a memory controller which connects to the system memory. A CPU and a GPU are connected by an interconnect such as PCIe or NVLINK.[](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/#gpu-cpu-system-diagram "Link to this image")
