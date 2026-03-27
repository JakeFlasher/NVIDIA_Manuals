---
title: "4.3.5.2. cudaPointerGetAttributes Query"
section: "4.3.5.2"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/stream-ordered-memory-allocation.html#cudapointergetattributes-query"
---

### [4.3.5.2. cudaPointerGetAttributes Query](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#cudapointergetattributes-query)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#cudapointergetattributes-query "Permalink to this headline")

Invoking `cudaPointerGetAttributes` on an allocation after invoking
`cudaFreeAsync` on it results in undefined behavior. Specifically, it does
not matter if an allocation is still accessible from a given stream: the
behavior is still undefined.
