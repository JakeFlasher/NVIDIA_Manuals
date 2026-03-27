---
title: "4.18.4.2.3. Synchronization"
section: "4.18.4.2.3"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/dynamic-parallelism.html#dynamic-parallelism--synchronization"
---

#### [4.18.4.2.3. Synchronization](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#synchronization)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#synchronization "Permalink to this headline")

It is up to the program to perform sufficient inter-thread synchronization, for example via a CUDA Event, if the calling thread is intended to synchronize with child grids invoked from other threads.

As it is not possible to explicitly synchronize child work from a parent thread, there is no way to guarantee that changes occurring in child grids are visible to threads within the parent grid.
