---
title: "2.4.2.2. Full Unified Memory Feature Support"
section: "2.4.2.2"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/understanding-memory.html#full-unified-memory-feature-support"
---

### [2.4.2.2. Full Unified Memory Feature Support](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics#full-unified-memory-feature-support)[](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/#full-unified-memory-feature-support "Permalink to this headline")

Most Linux systems have full unified memory support.  If device attribute `cudaDevAttrPageableMemoryAccess` is 1, then all system memory, whether allocated by CUDA APIs or system APIs, operates as unified memory with full feature support. This includes file-backed memory allocations created with `mmap`.

If `cudaDevAttrPageableMemoryAccess` is 0, then only memory allocated as managed memory by CUDA behaves as unified memory. Memory allocated with system APIs is not managed and is not necessarily accessible from GPU kernels.

In general, for unified allocations with full support:

- Managed memory is usually allocated in the memory space of the processor where it is first touched
- Managed memory is usually migrated when it is used by a processor other than the processor where it currently resides
- Managed memory is migrated or accessed at the granularity of memory pages (software coherence) or cache lines (hardware coherence)
- Oversubscription is allowed: an application may allocate more managed memory than is physically available on the GPU

Allocation and migration behavior can deviate from the above. This can by influenced the programmer using [hints and prefetches](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/#memory-mem-advise-prefetch).  Full coverage of full unified memory support can be found in [Unified Memory on Devices with Full CUDA Unified Memory Support](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/unified-memory.html#um-pageable-systems).
