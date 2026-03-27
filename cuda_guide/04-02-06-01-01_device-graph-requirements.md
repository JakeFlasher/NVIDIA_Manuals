---
title: "4.2.6.1.1. Device Graph Requirements"
section: "4.2.6.1.1"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html#device-graph-requirements"
---

#### [4.2.6.1.1. Device Graph Requirements](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#device-graph-requirements)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#device-graph-requirements "Permalink to this headline")

General requirements:

- The graph’s nodes must all reside on a single device.
- The graph can only contain kernel nodes, memcpy nodes, memset nodes, and child graph nodes.

Kernel nodes:

- Use of CUDA Dynamic Parallelism by kernels in the graph is not permitted.
- Cooperative launches are permitted so long as MPS is not in use.

Memcpy nodes:

- Only copies involving device memory and/or pinned device-mapped host memory are permitted.
- Copies involving CUDA arrays are not permitted.
- Both operands must be accessible from the current device at time of instantiation. Note that the copy operation will be performed from the device on which the graph resides, even if it is targeting memory on another device.
