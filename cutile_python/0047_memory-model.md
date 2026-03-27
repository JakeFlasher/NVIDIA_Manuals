---
title: "Memory Model"
section: ""
source: "https://docs.nvidia.com/cuda/cutile-python/memory_model.html#memory-model"
---

# [Memory Model](https://docs.nvidia.com/cuda/cutile-python#memory-model)[](https://docs.nvidia.com/cuda/cutile-python/#memory-model "Permalink to this headline")

cuTile employs a memory model that permits the compiler and hardware to reorder operations for performance.
As a result, without explicit synchronization, there is no guaranteed ordering of memory accesses across threads.

To coordinate memory accesses among threads, cuTile provides two attributes for atomic operations:

- **Memory Order**: Defines the memory ordering semantics of an atomic operation.
- **Memory Scope**: Defines the scope of threads that participate in memory ordering.

Synchronization occurs at a per-element granularity. Each element in the array participates independently in the memory model.

For a more detailed explanation, see the Memory Model section in the [Tile IR documentation](https://docs.nvidia.com/cuda/tile-ir/).
