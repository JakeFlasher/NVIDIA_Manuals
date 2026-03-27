---
title: "4.1.4.5. GPU Memory Oversubscription"
section: "4.1.4.5"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/unified-memory.html#gpu-memory-oversubscription"
---

### [4.1.4.5. GPU Memory Oversubscription](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#gpu-memory-oversubscription)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#gpu-memory-oversubscription "Permalink to this headline")

Unified memory enables applications to _oversubscribe_ the memory of any individual processor:
in other words they can allocate and share arrays larger than
the memory capacity of any individual processor in the system,
enabling among others out-of-core processing of datasets that do not fit within
a single GPU, without adding significant complexity to the programming model.

Additionally, multiple attributes can be queried by using corresponding `cudaMemRangeGetAttributes` function.
