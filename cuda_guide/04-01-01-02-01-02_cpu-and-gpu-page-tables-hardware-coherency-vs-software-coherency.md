---
title: "4.1.1.2.1.2. CPU and GPU Page Tables: Hardware Coherency vs. Software Coherency"
section: "4.1.1.2.1.2"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/unified-memory.html#cpu-and-gpu-page-tables-hardware-coherency-vs-software-coherency"
---

##### [4.1.1.2.1.2. CPU and GPU Page Tables: Hardware Coherency vs. Software Coherency](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#cpu-and-gpu-page-tables-hardware-coherency-vs-software-coherency)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#cpu-and-gpu-page-tables-hardware-coherency-vs-software-coherency "Permalink to this headline")

Hardware-coherent systems such as NVIDIA Grace Hopper offer a logically combined
page table for both CPUs and GPUs.
This is important because in order to access system-allocated memory from the GPU,
the GPU uses whichever page table entry was created by the CPU for the requested memory.
If that page table entry uses the default CPU page size of 4KiB or 64KiB,
accesses to large virtual memory areas will cause significant TLB misses,
thus significant slowdowns.

On the other hand, on software-coherent systems where the CPUs and GPUs each have their own logical
page table, different performance tuning aspects should be considered:
in order to guarantee coherency, these systems
usually use _page faults_ in case a processor accesses a memory address mapped
into the physical memory of a different processor. Such a page fault means that:

- It needs to be ensured that the currently owning processor (where the physical page currently resides)
cannot access this page anymore, either by deleting the page table entry or updating it.
- It needs to be ensured that the processor requesting access can access this page,
either by creating a new page table entry or updating and existing entry, such that
it becomes valid/active.
- The physical page backing this virtual page must be moved/migrated to the processor
requesting access: this can be an expensive operation, and the amount of work
is proportional to the page size.

Overall, hardware-coherent systems provide significant performance benefits
compared to software-coherent systems in cases where frequent concurrent accesses
to the same memory page are made by both CPU and GPU threads:

- less page-faults: these systems do not need to use page-faults for emulating coherency or migrating memory,
- less contention: these systems are coherent at cache-line granularity instead of page-size granularity, that is,
when there is contention from multiple processors within a cache line, only the cache line is exchanged which is much smaller than the smallest page-size,
and when the different processors access different cache-lines within a page, then there is no contention.

This impacts the performance of the following scenarios:

- atomic updates to the same address concurrently from both CPUs and GPUs
- signaling a GPU thread from a CPU thread or vice-versa.
