---
title: "4.3.3.4.5. IPC Import Pool Limitations"
section: "4.3.3.4.5"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/stream-ordered-memory-allocation.html#ipc-import-pool-limitations"
---

#### [4.3.3.4.5. IPC Import Pool Limitations](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#ipc-import-pool-limitations)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#ipc-import-pool-limitations "Permalink to this headline")

Allocating from an import pool is not allowed; specifically, import pools
cannot be set current and cannot be used in the `cudaMallocFromPoolAsync`
API. As such, the allocation reuse policy attributes do not have meaning for these
pools.

IPC Import pools, like IPC export pools, currently do not support releasing physical blocks back to the OS.

The resource usage stat attribute queries only reflect the allocations
imported into the process and the associated physical memory.
