---
title: "4.17.1.4. Memory management extensions to current APIs"
section: "4.17.1.4"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/extended-gpu-memory.html#memory-management-extensions-to-current-apis"
---

### [4.17.1.4. Memory management extensions to current APIs](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#memory-management-extensions-to-current-apis)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#memory-management-extensions-to-current-apis "Permalink to this headline")

Currently, EGM memory can be mapped with Virtual Memory (`cuMemCreate`)  or
Stream Ordered Memory (`cudaMemPoolCreate`) allocators. The user is
responsible for allocating physical memory and mapping it to a virtual
memory address space on all sockets.

> **Note**
>
> Multi-node, multi-GPU platforms require interprocess
> communication. Therefore we encourage the reader to see [Chapter 4.15](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/inter-process-communication.html#interprocess-communication).

> **Note**
>
> We encourage readers to read CUDA Programming Guide’s [Chapter 4.16](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/virtual-memory-management.html#virtual-memory-management) and [Chapter 4.3](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/stream-ordered-memory-allocation.html#stream-ordered-memory-allocator) for a better understanding.

New CUDA property types have been added to APIs for allowing those
approaches to understand allocation locations using NUMA-like node
identifiers:

| **CUDA Type** | **Used with** |
| --- | --- |
| `CU_MEM_LOCATION_TYPE_HOST_NUMA` | `CUmemAllocationProp` for `cuMemCreate` |
| `cudaMemLocationTypeHostNuma` | `cudaMemPoolProps` for `cudaMemPoolCreate` |

> **Note**
>
> Please see  [CUDA Driver API](https://www.google.com/url?q=https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html&sa=D&source=editors&ust=1696873412599124&usg=AOvVaw0Ru93Acs_FpJG0gl02BLMX)
> and [CUDA Runtime Data Types](https://www.google.com/url?q=https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html%23group__CUDART__TYPES_1gg2279aa08666f329f3ba4afe397fa60f024dc63fb938dee27b41e3842da35d2d0&sa=D&source=editors&ust=1696873412599344&usg=AOvVaw2O-SyvDt1G37IjcpFzc-4C)
> to find more about NUMA specific CUDA types.
