---
title: "4.16.4.1. Allocating Multicast Objects"
section: "4.16.4.1"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/virtual-memory-management.html#allocating-multicast-objects"
---

### [4.16.4.1. Allocating Multicast Objects](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#allocating-multicast-objects)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#allocating-multicast-objects "Permalink to this headline")

Multicast objects can be created with `cuMulticastCreate`:

```c++
CUmemGenericAllocationHandle createMCHandle(int numDevices, size_t size) {
    CUmemAllocationProp mcProp = {};
    mcProp.numDevices = numDevices;
    mcProp.handleTypes = CU_MEM_HANDLE_TYPE_FABRIC; // or on single node CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR

    size_t granularity = 0;
    cuMulticastGetGranularity(&granularity, &mcProp, CU_MEM_ALLOC_GRANULARITY_MINIMUM);

    // Ensure size matches granularity requirements for the allocation
    size_t padded_size = ROUND_UP(size, granularity);

    mcProp.size = padded_size;

    // Create Multicast Object this has no devices and no physical memory associated yet
    CUmemGenericAllocationHandle mcHandle;
    cuMulticastCreate(&mcHandle, &mcProp);

    return mcHandle;
}
```
