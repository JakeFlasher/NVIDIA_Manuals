---
title: "4.3.5.3. cudaGraphAddMemsetNode"
section: "4.3.5.3"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/stream-ordered-memory-allocation.html#cudagraphaddmemsetnode"
---

### [4.3.5.3. cudaGraphAddMemsetNode](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#cudagraphaddmemsetnode)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#cudagraphaddmemsetnode "Permalink to this headline")

`cudaGraphAddMemsetNode` does not work with memory allocated via the stream
ordered allocator. However, memsets of the allocations can be stream captured.
