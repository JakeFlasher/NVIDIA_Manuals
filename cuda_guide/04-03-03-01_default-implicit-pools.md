---
title: "4.3.3.1. Default/Implicit Pools"
section: "4.3.3.1"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/stream-ordered-memory-allocation.html#default-implicit-pools"
---

### [4.3.3.1. Default/Implicit Pools](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#default-implicit-pools)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#default-implicit-pools "Permalink to this headline")

The default memory pool of a device can be retrieved by calling
`cudaDeviceGetDefaultMempool`. Allocations from the default memory pool
of a device are non-migratable device allocation located on that device. These
allocations will always be accessible from that device. The accessibility of
the default memory pool can be modified with `cudaMemPoolSetAccess` and
queried with `cudaMemPoolGetAccess`. Since the default pools do not need to be
explicitly created, they are sometimes referred to as implicit pools. The
default memory pool of a device does not support IPC.
