---
title: "4.17.2.2.1. Using VMM APIs"
section: "4.17.2.2.1"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/extended-gpu-memory.html#using-vmm-apis"
---

#### [4.17.2.2.1. Using VMM APIs](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#using-vmm-apis)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#using-vmm-apis "Permalink to this headline")

The first step in memory allocation using Virtual Memory Management APIs
is to create a physical memory chunk that will provide a backing for the
allocation. See CUDA Programming Guide’s [Virtual Memory Management section](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#virtual-memory-management)
for more details. In EGM allocations the user has to explicitly provide
`CU_MEM_LOCATION_TYPE_HOST_NUMA`  as the location type and
*numaID* as the location identifier. Also in EGM, allocations must
be aligned to appropriate granularity of the platform. The following
code snippet shows allocating physical memory with `cuMemCreate`:

```c++
CUmemAllocationProp prop{};
prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
prop.location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA;
prop.location.id = numaId;
size_t granularity = 0;
cuMemGetAllocationGranularity(&granularity, &prop, MEM_ALLOC_GRANULARITY_MINIMUM);
size_t padded_size = ROUND_UP(size, granularity);
CUmemGenericAllocationHandle allocHandle;
cuMemCreate(&allocHandle, padded_size, &prop, 0);
```

After physical memory allocation, we have to reserve an address space
and map it to a pointer. These procedures do not have EGM-specific
changes:

```c++
CUdeviceptr dptr;
cuMemAddressReserve(&dptr, padded_size, 0, 0, 0);
cuMemMap(dptr, padded_size, 0, allocHandle, 0);
```

Finally, the user has to explicitly protect mapped virtual address
ranges. Otherwise access to the mapped space would result in a crash.
Similar to the memory allocation, the user has to provide
`CU_MEM_LOCATION_TYPE_HOST_NUMA` as the location type and
*numaId* as the location identifier. Following code snippet create
an access descriptors for the host node and the GPU to give read and
write access for the mapped memory to both of them:

```c++
CUmemAccessDesc accessDesc[2]{{}};
accessDesc[0].location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA;
accessDesc[0].location.id = numaId;
accessDesc[0].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
accessDesc[1].location.type = CU_MEM_LOCATION_TYPE_DEVICE;
accessDesc[1].location.id = currentDev;
accessDesc[1].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
cuMemSetAccess(dptr, size, accessDesc, 2);
```
