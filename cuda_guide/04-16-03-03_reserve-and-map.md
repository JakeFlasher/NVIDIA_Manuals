---
title: "4.16.3.3. Reserve and Map"
section: "4.16.3.3"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/virtual-memory-management.html#reserve-and-map"
---

### [4.16.3.3. Reserve and Map](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#reserve-and-map)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#reserve-and-map "Permalink to this headline")

**Reserving a Virtual Address Range**

Since notions of address and memory are distinct in VMM,
applications must carve out an address range that can hold the
memory allocations made by `cuMemCreate`. The address range reserved must be
at least as large as the sum of the sizes of all the physical memory
allocations the user plans to place in them.

Applications can reserve a virtual address range by passing appropriate
parameters to `cuMemAddressReserve`. The address range obtained will not
have any device or host physical memory associated with it. The reserved
virtual address range can be mapped to memory chunks belonging to any device
in the system, thus providing the application a continuous VA range backed and
mapped by memory belonging to different devices. Applications are expected to
return the virtual address range back to CUDA using `cuMemAddressFree`.
Users must ensure that the entire VA range is unmapped before calling
`cuMemAddressFree`. These functions are conceptually similar to `mmap` and `munmap`
on Linux or `VirtualAlloc` AND `VirtualFree` on Windows. The following
code snippet illustrates the usage for the function:

```c++
CUdeviceptr ptr;
// `ptr` holds the returned start of virtual address range reserved.
CUresult result = cuMemAddressReserve(&ptr, size, 0, 0, 0); // alignment = 0 for default alignment
```

**Mapping Memory**

The allocated physical memory and the carved out virtual address space from
the previous two sections represent the memory and address distinction
introduced by the VMM APIs. For the allocated memory to
be useable, the user must map the memory to the address space. The
address range obtained from `cuMemAddressReserve` and the physical
allocation obtained from `cuMemCreate` or `cuMemImportFromShareableHandle`
must be associated with each other by using `cuMemMap`.

Users can associate allocations from multiple devices to reside in contiguous
virtual address ranges as long as they have carved out enough address space.
To decouple the physical allocation and the address range, users must
unmap the address of the mapping with `cuMemUnmap`. Users can map and
unmap memory to the same address range as many times as they want, so long as
they ensure that they don’t attempt to create mappings on VA range
reservations that are already mapped. The following code snippet illustrates
the usage for the function:

```c++
CUdeviceptr ptr;
// `ptr`: address in the address range previously reserved by cuMemAddressReserve.
// `allocHandle`: CUmemGenericAllocationHandle obtained by a previous call to cuMemCreate.
CUresult result = cuMemMap(ptr, size, 0, allocHandle, 0);
```
