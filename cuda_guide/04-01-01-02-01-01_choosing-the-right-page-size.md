---
title: "4.1.1.2.1.1. Choosing the Right Page Size"
section: "4.1.1.2.1.1"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/unified-memory.html#choosing-the-right-page-size"
---

##### [4.1.1.2.1.1. Choosing the Right Page Size](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#choosing-the-right-page-size)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#choosing-the-right-page-size "Permalink to this headline")

In general, small page sizes lead to less (virtual) memory fragmentation but more TLB misses,
whereas larger page sizes lead to more memory fragmentation but less TLB misses.
Additionally, memory migration is generally more expensive with larger page sizes compared to
smaller page sizes, because we typically migrate full memory pages. This can cause
larger latency spikes in an application using large page sizes. See also the next section
for more details on page faults.

One important aspect for performance tuning is that TLB misses are generally
significantly more expensive on the GPU compared to the CPU. This means that
if a GPU thread frequently accesses random locations of unified memory mapped
using a small enough page size, it might be significantly slower compared to
the same accesses to unified memory mapped using a large enough page size.
While a similar effect might occur for a CPU thread randomly accessing a large
area of memory mapped using a small page size, the slowdown is less pronounced,
meaning that the application might want to trade-off this slowdown with
having less memory fragmentation.

Note that in general, applications should not tune their performance to the
physical page size of a given processor, since physical page sizes are subject
to change depending on the hardware. The advice above only applies to virtual
page sizes.
