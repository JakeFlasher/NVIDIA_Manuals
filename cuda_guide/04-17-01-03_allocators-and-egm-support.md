---
title: "4.17.1.3. Allocators and EGM support"
section: "4.17.1.3"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/extended-gpu-memory.html#allocators-and-egm-support"
---

### [4.17.1.3. Allocators and EGM support](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#allocators-and-egm-support)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#allocators-and-egm-support "Permalink to this headline")

Mapping system memory as EGM does not cause any performance issues. In
fact, accessing a remote socket’s system memory mapped as EGM is going
to be faster. Because, with EGM traffic is guaranteed to be routed over
NVLinks. Currently, `cuMemCreate` and `cudaMemPoolCreate` allocators are
supported with appropriate location type and NUMA identifiers.
