---
title: "4.3.4.4. Memory Reuse Policies"
section: "4.3.4.4"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/stream-ordered-memory-allocation.html#memory-reuse-policies"
---

### [4.3.4.4. Memory Reuse Policies](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#memory-reuse-policies)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#memory-reuse-policies "Permalink to this headline")

In order to service an allocation request, the driver attempts to reuse memory
that was previously freed via `cudaFreeAsync()` before attempting to
allocate more memory from the OS. For example, memory freed in a stream can
be reused immediately in a subsequent allocation request on the same stream.
When a stream is synchronized with the CPU, the memory that was
previously freed in that stream becomes available for reuse for an allocation
in any stream.
Reuse policies can be applied to both default and explicit memory pools.

The stream-ordered allocator has a few controllable allocation policies. The
pool attributes `cudaMemPoolReuseFollowEventDependencies`,
`cudaMemPoolReuseAllowOpportunistic`, and
`cudaMemPoolReuseAllowInternalDependencies` control these policies and are detailed below.
These policies can be enabled or disabled  through a call to `cudaMemPoolSetAttribute`.
Upgrading to a newer CUDA driver may change, enhance, augment and/or reorder
the enumeration of the reuse policies.
