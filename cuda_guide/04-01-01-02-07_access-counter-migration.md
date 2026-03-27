---
title: "4.1.1.2.7. Access Counter Migration"
section: "4.1.1.2.7"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/unified-memory.html#access-counter-migration"
---

#### [4.1.1.2.7. Access Counter Migration](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#access-counter-migration)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#access-counter-migration "Permalink to this headline")

On hardware-coherent systems, the access counters feature keeps track of the frequency of access that a GPU makes to memory located on other processors. This is needed to ensure memory pages are moved to the physical memory of the processor that is accessing the pages most frequently. It can guide migrations between CPU and GPU, as well as between peer GPUs, a process called access counter migration.

Starting with CUDA 12.4, access counters are supported system-allocated memory. Note that file-backed memory does not migrate based on access. For system-allocated memory, access counters migration can be switched on by using the `cudaMemAdviseSetAccessedBy` hint to a device with the corresponding device id. If access counters are on, one can use  `cudaMemAdviseSetPreferredLocation` set to host to prevent migrations.  Per default `cudaMallocManaged` migrates based on a fault-and-migrate mechanism. [^[7]]

The driver may also use access counters for more efficient thrashing mitigation or memory oversubscription scenarios.

[^[7]]: Current systems allow the use of access-counter migration with managed memory when the accessed-by device hint is set. This is an implementation detail and should not be relied on for future compatibility.
