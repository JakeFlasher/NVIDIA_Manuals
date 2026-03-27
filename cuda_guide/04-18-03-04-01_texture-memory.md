---
title: "4.18.3.4.1. Texture Memory"
section: "4.18.3.4.1"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/dynamic-parallelism.html#texture-memory"
---

#### [4.18.3.4.1. Texture Memory](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#texture-memory)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#texture-memory "Permalink to this headline")

Writes to the global memory region over which a texture is mapped are incoherent with respect to texture accesses. Coherence for texture memory is enforced at the invocation of a child grid and when a child grid completes. This means that writes to memory prior to a child kernel launch are reflected in texture memory accesses of the child. Similarly to Global Memory above, writes to memory by a child are never guaranteed to be reflected in the texture memory accesses by a parent. The only way to access the modifications made by the threads in the child grid before the parent grid exits is via a kernel launched into the `cudaStreamTailLaunch` stream. Concurrent accesses by parent and child may result in inconsistent data.
