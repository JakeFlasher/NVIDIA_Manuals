---
title: "4.3.3.4. Enabling Memory Pools for IPC"
section: "4.3.3.4"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/stream-ordered-memory-allocation.html#enabling-memory-pools-for-ipc"
---

### [4.3.3.4. Enabling Memory Pools for IPC](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#enabling-memory-pools-for-ipc)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#enabling-memory-pools-for-ipc "Permalink to this headline")

Memory pools can be enabled for interprocess communication (IPC) to
allow easy, efficient and secure sharing of GPU
memory between processes. CUDA’s IPC memory pools provide the same security
benefits as CUDA’s [virtual memory management APIs](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#virtual-memory-management).

There are two steps to sharing memory between processes with memory pools:
the processes first needs to share access to the pool, then share specific
allocations from that pool. The first step establishes and enforces security.
The second step coordinates what virtual addresses are used in each process
and when mappings need to be valid in the importing process.
