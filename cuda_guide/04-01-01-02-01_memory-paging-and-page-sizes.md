---
title: "4.1.1.2.1. Memory Paging and Page Sizes"
section: "4.1.1.2.1"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/unified-memory.html#memory-paging-and-page-sizes"
---

#### [4.1.1.2.1. Memory Paging and Page Sizes](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#memory-paging-and-page-sizes)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#memory-paging-and-page-sizes "Permalink to this headline")

To better understand the performance implication of unified memory, it is important to understand virtual addressing,
memory pages and page sizes.
This sub-section attempts to define all necessary terms and explain why paging matters for performance.

All currently supported systems for unified memory use a virtual address space:
this means that memory addresses used by an application represent a _virtual_ location
which might be _mapped_ to a physical location where the memory actually resides.

All currently supported processors, including both CPUs and GPUs, additionally use
memory _paging_. Because all systems use a virtual address space, there are two types
of memory pages:

- Virtual pages: This represents a fixed-size contiguous chunk of virtual memory
per process tracked by the operating system, which can be _mapped_ into physical memory.
Note that the virtual page is linked to the _mapping_: for example, a single
virtual address might be mapped into physical memory using different page sizes.
- Physical pages: This represents a fixed-size contiguous chunk of memory
the processor’s main Memory Management Unit (MMU) supports and into which
a virtual page can be mapped.

Currently, all x86_64 CPUs use a default physical page size of 4KiB.
Arm CPUs support multiple physical page sizes - 4KiB, 16KiB, 32KiB and 64KiB - depending on the exact CPU.
Finally, NVIDIA GPUs support multiple physical page sizes, but prefer 2MiB physical pages or larger.
Note that these sizes are subject to change in future hardware.

The default page size of virtual pages usually corresponds to the physical page size,
but an application may use different page sizes as long as they are supported by the
operating system and the hardware. Typically, supported virtual page sizes must be
powers of 2 and multiples of the physical page size.

The logical entity tracking the mapping of virtual pages into physical pages will be referred to as a _page table_,
and each mapping of a given virtual page with a given virtual size to physical pages is called a _Page Table Entry (PTE)_.
All supported processors provide specific caches for the page table to speed up the translation of
virtual addresses to physical addresses. These caches are called _Translation Lookaside Buffers (TLBs)_.

There are two important aspects for performance tuning of applications:

- the choice of virtual page size,
- whether the system offers a combined page table used by both CPUs and GPUs,
or separate page tables for each CPU and GPU individually.
