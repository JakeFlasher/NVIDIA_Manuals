---
title: "4.3.5.4. Pointer Attributes"
section: "4.3.5.4"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/stream-ordered-memory-allocation.html#pointer-attributes"
---

### [4.3.5.4. Pointer Attributes](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#pointer-attributes)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#pointer-attributes "Permalink to this headline")

The `cudaPointerGetAttributes` query works on stream-ordered allocations.
Since stream-ordered allocations are not context associated, querying
`CU_POINTER_ATTRIBUTE_CONTEXT` will succeed but return NULL in `*data`.
The attribute `CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL` can be used to determine
the location of the allocation: this can be useful when selecting a context
for making p2h2p copies using `cudaMemcpyPeerAsync`. The attribute
`CU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE` was added in CUDA 11.3 and can be
useful for debugging and for confirming which pool an allocation comes from
before doing IPC.
