---
title: "4.18.3. Memory Coherence and Consistency"
section: "4.18.3"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/dynamic-parallelism.html#memory-coherence-and-consistency"
---

## [4.18.3. Memory Coherence and Consistency](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#memory-coherence-and-consistency)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#memory-coherence-and-consistency "Permalink to this headline")

Parent and child grids share the same global and constant memory storage, but have distinct local and shared memory. The following table shows which memory spaces allow parent and child to access via the same pointers. Child grids can never access the local or shared memory of parent grids, nor can parent grids access local or shared memory of child grids.

| Memory Space | Parent/Child use same pointers? |
| --- | --- |
| Global Memory | Yes |
| Mapped Memory | Yes |
| Local Memory | No |
| Shared Memory | No |
| Texture Memory | Yes (read-only) |
