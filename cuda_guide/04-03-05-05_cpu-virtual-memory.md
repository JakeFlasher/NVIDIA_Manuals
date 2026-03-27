---
title: "4.3.5.5. CPU Virtual Memory"
section: "4.3.5.5"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/stream-ordered-memory-allocation.html#cpu-virtual-memory"
---

### [4.3.5.5. CPU Virtual Memory](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#cpu-virtual-memory)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#cpu-virtual-memory "Permalink to this headline")

When using CUDA stream-ordered memory allocator APIs, avoid setting VRAM limitations with “ulimit -v” as this is not supported.
