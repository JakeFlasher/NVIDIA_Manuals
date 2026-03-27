---
title: "4.2.5.3.2. Physical Memory Management and Sharing"
section: "4.2.5.3.2"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html#physical-memory-management-and-sharing"
---

#### [4.2.5.3.2. Physical Memory Management and Sharing](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#physical-memory-management-and-sharing)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#physical-memory-management-and-sharing "Permalink to this headline")

CUDA is responsible for mapping physical memory to the virtual address before the allocating node is reached in GPU order. As an optimization for memory footprint and mapping overhead, multiple graphs may use the same physical memory for distinct allocations if they will not run simultaneously; however, physical pages cannot be reused if they are bound to more than one executing graph at the same time, or to a graph allocation which remains unfreed.

CUDA may update physical memory mappings at any time during graph instantiation, launch, or execution. CUDA may also introduce synchronization between future graph launches in order to prevent live graph allocations from referring to the same physical memory. As for any allocate-free-allocate pattern, if a program accesses a pointer outside of an allocation’s lifetime, the erroneous access may silently read or write live data owned by another allocation (even if the virtual address of the allocation is unique). Use of compute sanitizer tools can catch this error.

The following figure shows graphs sequentially launched in the same stream. In this example, each graph frees all the memory it allocates. Since the graphs in the same stream never run concurrently, CUDA can and should use the same physical memory to satisfy all the allocations.

![Sequentially Launched Graphs](images/________-______-__________-___-________1.png)

Figure 30 Sequentially Launched Graphs[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#id11 "Link to this image")
