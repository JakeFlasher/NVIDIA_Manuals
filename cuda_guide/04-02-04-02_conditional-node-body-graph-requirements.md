---
title: "4.2.4.2. Conditional Node Body Graph Requirements"
section: "4.2.4.2"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html#conditional-node-body-graph-requirements"
---

### [4.2.4.2. Conditional Node Body Graph Requirements](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#conditional-node-body-graph-requirements)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#conditional-node-body-graph-requirements "Permalink to this headline")

General requirements:

- The graph’s nodes must all reside on a single device.
- The graph can only contain kernel nodes, empty nodes, memcpy nodes, memset nodes, child graph nodes, and conditional nodes.

Kernel nodes:

- Use of CUDA Dynamic Parallelism or Device Graph Launch by kernels in the graph is not permitted.
- Cooperative launches are permitted so long as MPS is not in use.

Memcpy/Memset nodes:

- Only copies/memsets involving device memory and/or pinned device-mapped host memory are permitted.
- Copies/memsets involving CUDA arrays are not permitted.
- Both operands must be accessible from the current device at time of instantiation. Note that the copy operation will be performed from the device on which the graph resides, even if it is targeting memory on another device.
