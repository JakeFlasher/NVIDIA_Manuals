---
title: "4.16.3. Unicast Memory Sharing"
section: "4.16.3"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/virtual-memory-management.html#unicast-memory-sharing"
---

## [4.16.3. Unicast Memory Sharing](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#unicast-memory-sharing)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#unicast-memory-sharing "Permalink to this headline")

Sharing GPU memory can happen on one machine with multiple GPUs or across a
network of machines. The process follows these steps:

- Allocate and Export: A CUDA program on one GPU allocates memory and gets a
sharable handle for it.
- Share and Import: The handle is then sent to other programs on the node
using IPC, MPI, or NCCL etc. In the receiving GPUs, the CUDA driver imports
the handle, creates the necessary memory objects
- Reserve and Map: The driver creates a mapping from the program’s Virtual
Address (VA) to the GPU’s Physical Address (PA) to its network Fabric
Address (FA).
- Access Rights: Setting access rights for the allocation.
- Releasing the Memory: Freeing all allocations when program ends its execution.

![Unicast Memory Sharing Example](images/_______-______-________1.png)

Figure 53 Unicast Memory Sharing Example[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#unicast-memory-sharing-example "Link to this image")
