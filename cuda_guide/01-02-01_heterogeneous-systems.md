---
title: "1.2.1. Heterogeneous Systems"
section: "1.2.1"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html#heterogeneous-systems"
---

## [1.2.1. Heterogeneous Systems](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction#heterogeneous-systems)[](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/#heterogeneous-systems "Permalink to this headline")

The CUDA programming model assumes a heterogeneous computing system, which means a system that includes both GPUs and CPUs. The CPU and the memory directly connected to it are called the _host_ and _host memory_, respectively. A GPU and the memory directly connected to it are referred to as the _device_ and _device memory_, respectively. In some system-on-chip (SoC) systems, these may be part of a single package. In larger systems, there may be multiple CPUs or GPUs.

CUDA applications execute some part of their code on the GPU, but applications always start execution on the CPU. The host code, which is the code that runs on the CPU, can use CUDA APIs to copy data between the host memory and device memory, start code executing on the GPU, and wait for data copies or GPU code to complete. The CPU and GPU can both be executing code simultaneously, and best performance is usually found by maximizing utilization of both CPUs and GPUs.

The code an application executes on the GPU is referred to as _device code_, and a function that is invoked for execution on the GPU is, for historical reasons, called a _kernel_. The act of starting a kernel running is called _launching_ the kernel. A kernel launch can be thought of as starting many threads executing the kernel code in parallel on the GPU. GPU threads operate similarly to threads on CPUs, though there are some differences important to both correctness and performance that will be covered in later sections (see [Section 3.2.2.1.1](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html#advanced-kernels-independent-thread-scheduling)).
