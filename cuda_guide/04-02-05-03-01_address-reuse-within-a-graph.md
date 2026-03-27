---
title: "4.2.5.3.1. Address Reuse within a Graph"
section: "4.2.5.3.1"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html#address-reuse-within-a-graph"
---

#### [4.2.5.3.1. Address Reuse within a Graph](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#address-reuse-within-a-graph)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#address-reuse-within-a-graph "Permalink to this headline")

CUDA may reuse memory within a graph by assigning the same virtual address ranges to different allocations whose lifetimes do not overlap. Since virtual addresses may be reused, pointers to different allocations with disjoint lifetimes are not guaranteed to be unique.

The following figure shows adding a new allocation node (2) that can reuse the address freed by a dependent node (1).

![Adding New Alloc Node 2](images/_______-_____-w_____-_-______1.png)

Figure 28 Adding New Alloc Node 2[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#id9 "Link to this image")

The following figure shows adding a new alloc node (4). The new alloc node is not dependent on the free node (2) so cannot reuse the address from the associated alloc node (2). If the alloc node (2) used the address freed by free node (1), the new alloc node 3 would need a new address.

![Adding New Alloc Node 3](images/_______-_____-w_____-_-______2.png)

Figure 29 Adding New Alloc Node 3[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#id10 "Link to this image")
