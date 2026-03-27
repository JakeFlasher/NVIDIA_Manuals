---
title: "3.5.3.2. Dynamic Parallelism"
section: "3.5.3.2"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/feature-survey.html#dynamic-parallelism"
---

### [3.5.3.2. Dynamic Parallelism](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced#dynamic-parallelism)[](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/#dynamic-parallelism "Permalink to this headline")

CUDA applications most commonly launch kernels from code running on the CPU. It is also possible to create new kernel invocations from a kernel running on the GPU. This feature is referred to as [CUDA dynamic parallelism](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/dynamic-parallelism.html#cuda-dynamic-parallelism). [Section 4.18](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/dynamic-parallelism.html#cuda-dynamic-parallelism) covers the details of creating new GPU kernel launches from code running on the GPU.
