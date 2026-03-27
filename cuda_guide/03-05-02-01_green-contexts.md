---
title: "3.5.2.1. Green Contexts"
section: "3.5.2.1"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/feature-survey.html#green-contexts"
---

### [3.5.2.1. Green Contexts](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced#green-contexts)[](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/#green-contexts "Permalink to this headline")

[Green contexts](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/green-contexts.html#green-contexts), also called _execution contexts_,  is the name given to a CUDA feature which enables a program to create [CUDA contexts](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/driver-api.html#driver-api-context) which will execute work only on a subset of the SMs of a GPU. By default, the thread blocks of a kernel launch are dispatched to any SM within the GPU which can fulfill the resource requirements of the kernel. There are a large number of factors which can affect which SMs can execute a thread block, including but not necessarily limited to: shared memory use, register use, use of clusters, and total number of threads in the thread block.

Execution contexts allow a kernel to be launched into a specially created context which further limits the number of SMs available to execute the kernel. Importantly, when a program creates a green context which uses some set of SMs, other contexts on the GPU will not schedule thread blocks onto the SMs allocated to the green context. This includes the primary context, which is the default context used by the CUDA runtime. This allows these SMs to be reserved for workloads which are high priority or latency-sensitive.

[Section 4.6](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/green-contexts.html#green-contexts) gives full details on the use of green contexts. Green contexts are available in the CUDA runtime in CUDA 13.1 and later.
