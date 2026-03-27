---
title: "4.2.5.3. Optimized Memory Reuse"
section: "4.2.5.3"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html#optimized-memory-reuse"
---

### [4.2.5.3. Optimized Memory Reuse](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#optimized-memory-reuse)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#optimized-memory-reuse "Permalink to this headline")

CUDA reuses memory in two ways:

- Virtual and physical memory reuse within a graph is based on virtual address assignment, like in the stream ordered allocator.
- Physical memory reuse between graphs is done with virtual aliasing: different graphs can map the same physical memory to their unique virtual addresses.
