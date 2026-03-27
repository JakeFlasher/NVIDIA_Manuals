---
title: "4.3.4.5. Synchronization API Actions"
section: "4.3.4.5"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/stream-ordered-memory-allocation.html#synchronization-api-actions"
---

### [4.3.4.5. Synchronization API Actions](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#synchronization-api-actions)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#synchronization-api-actions "Permalink to this headline")

One of the optimizations that comes with the allocator being part of the CUDA
driver is integration with the synchronize APIs. When the user requests that
the CUDA driver synchronize, the driver waits for asynchronous work to
complete. Before returning, the driver will determine what frees the
synchronization guaranteed to be completed. These allocations are made
available for allocation regardless of specified stream or disabled allocation
policies. The driver also checks `cudaMemPoolAttrReleaseThreshold` here and
releases any excess physical memory that it can.
