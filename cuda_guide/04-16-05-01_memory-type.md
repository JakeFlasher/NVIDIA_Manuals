---
title: "4.16.5.1. Memory Type"
section: "4.16.5.1"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/virtual-memory-management.html#memory-type"
---

### [4.16.5.1. Memory Type](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#memory-type)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#memory-type "Permalink to this headline")

VMM also provides a mechanism for applications to allocate special types of memory that
certain devices may support. With `cuMemCreate`,
applications can  specify memory type requirements using the
`CUmemAllocationProp::allocFlags` to opt-in to specific memory features.
Applications must ensure that the requested memory type is supported by
the device.
