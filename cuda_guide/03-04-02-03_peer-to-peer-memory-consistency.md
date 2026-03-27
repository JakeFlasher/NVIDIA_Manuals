---
title: "3.4.2.3. Peer-to-Peer Memory Consistency"
section: "3.4.2.3"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/multi-gpu-systems.html#peer-to-peer-memory-consistency"
---

### [3.4.2.3. Peer-to-Peer Memory Consistency](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced#peer-to-peer-memory-consistency)[](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/#peer-to-peer-memory-consistency "Permalink to this headline")

Synchronization operations must be used to enforce the ordering and
correctness of memory accesses by concurrently executing threads in grids
distributed across multiple devices.
Threads synchronizing across devices operate at the `thread_scope_system`
[synchronization scope](https://nvidia.github.io/cccl/libcudacxx/extended_api/memory_model.html#thread-scopes).
Similarly, memory operations fall within
the `thread_scope_system` [memory synchronization domain](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#memory-synchronization-domains).

CUDA ref::*atomic-functions* can perform read-modify-write operations
on an object in peer device memory when only a single GPU is
accessing that object.
The requirements and limitations for peer atomicity are described in the CUDA memory model [atomicity requirements](https://nvidia.github.io/cccl/libcudacxx/extended_api/memory_model.html#atomicity) discussion.
