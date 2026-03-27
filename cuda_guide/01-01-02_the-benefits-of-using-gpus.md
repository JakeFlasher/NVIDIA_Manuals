---
title: "1.1.2. The Benefits of Using GPUs"
section: "1.1.2"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/introduction.html#the-benefits-of-using-gpus"
---

## [1.1.2. The Benefits of Using GPUs](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction#the-benefits-of-using-gpus)[](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/#the-benefits-of-using-gpus "Permalink to this headline")

A GPU provides much higher instruction throughput and memory bandwidth than a CPU within a similar price and power envelope. Many applications leverage these capabilities to run significantly faster on the GPU than on the CPU (see [GPU Applications](https://www.nvidia.com/en-us/accelerated-applications/)). Other computing devices, like FPGAs, are also very energy efficient, but offer much less programming flexibility than GPUs.

GPUs and CPUs are designed with different goals in mind. While a CPU is designed to excel at executing a serial sequence of operations (called a thread) as fast as possible and can execute a few tens of these threads in parallel, a GPU is designed to excel at executing thousands of threads in parallel, trading off lower single-thread performance to achieve much greater total throughput.

GPUs are specialized for highly parallel computations and devote more transistors to data processing units, while CPUs dedicate more transistors to data caching and flow control. [Figure 1](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/#from-graphics-processing-to-general-purpose-parallel-computing-gpu-devotes-more-transistors-to-data-processing) shows an example distribution of chip resources for a CPU versus a GPU.

![The GPU Devotes More Transistors to Data Processing](images/___-________-__-_____-_____1.png)

Figure 1 The GPU Devotes More Transistors to Data Processing[](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/#from-graphics-processing-to-general-purpose-parallel-computing-gpu-devotes-more-transistors-to-data-processing "Link to this image")
