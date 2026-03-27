---
title: "4.1.4. Performance Hints"
section: "4.1.4"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/unified-memory.html#performance-hints"
---

## [4.1.4. Performance Hints](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#performance-hints)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#performance-hints "Permalink to this headline")

Performance hints allow programmers to provide CUDA with more information about unified memory usage.
CUDA uses performance hints to managed memory more efficiently and improve application performance.
Performance hints never impact the correctness of an application.
Performance hints only affect performance.

> **Note**
>
> Applications should only use unified memory performance hints if they improve performance.

Performance hints may be used on any unified memory allocation, including CUDA managed memory.
On systems with full CUDA unified memory support, performance hints can be applied to all system-allocated memory.
