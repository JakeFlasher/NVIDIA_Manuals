---
title: "1.2.3.2. On-Chip Memory in GPUs"
section: "1.2.3.2"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html#on-chip-memory-in-gpus"
---

### [1.2.3.2. On-Chip Memory in GPUs](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction#on-chip-memory-in-gpus)[](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/#on-chip-memory-in-gpus "Permalink to this headline")

In addition to the global memory, each GPU has some on-chip memory. Each SM has its own register file and shared memory. These memories are part of the SM and can be accessed extremely quickly from threads executing within the SM, but they are not accessible to threads running in other SMs.

The register file stores thread local variables which are usually allocated by the compiler. The shared memory is accessible by all threads within a thread block or cluster. Shared memory can be used for exchanging data between threads of a thread block or cluster.

The register file and unified data cache in an SM have finite sizes. The size of an SM’s register file, unified data cache, and how the unified data cache can be configured for L1 and shared memory balance can be found in [Memory Information per Compute Capability](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html#compute-capabilities-table-memory-information-per-compute-capability). The register file, shared memory space, and L1 cache are shared among all threads in a thread block.

To schedule a thread block to an SM, the total number of registers needed for each thread multiplied by the number of threads in the thread block must be less than or equal to the available registers in the SM. If the number of registers required for a thread block exceeds the size of the register file, the kernel is not launchable and the number of threads in the thread block must be decreased to make the thread block launchable.

Shared memory allocations are done at the thread block level. That is, unlike register allocations which are per thread, allocations of shared memory are common to the entire thread block.
