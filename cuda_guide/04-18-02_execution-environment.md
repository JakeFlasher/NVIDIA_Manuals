---
title: "4.18.2. Execution Environment"
section: "4.18.2"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/dynamic-parallelism.html#execution-environment"
---

## [4.18.2. Execution Environment](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#execution-environment)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#execution-environment "Permalink to this headline")

Dynamic parallelism in CUDA allows GPU threads to configure, launch, and implicitly synchronize new grids. A grid is an instance of a kernel launch, including the specific shape of the thread blocks and the grid of thread blocks. The distinction between a kernel function itself and the specific invocation of that kernel, i.e. a grid, is important to note in the following sections.
