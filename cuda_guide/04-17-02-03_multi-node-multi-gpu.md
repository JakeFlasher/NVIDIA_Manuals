---
title: "4.17.2.3. Multi-Node, Multi-GPU"
section: "4.17.2.3"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/extended-gpu-memory.html#multi-node-multi-gpu"
---

### [4.17.2.3. Multi-Node, Multi-GPU](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#multi-node-multi-gpu)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#multi-node-multi-gpu "Permalink to this headline")

Beyond memory allocation, remote peer access does not have EGM-specific
modification and it follows CUDA inter process (IPC) protocol. See
[CUDA Programming Guide](https://www.google.com/url?q=https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html%23allocating-physical-memory&sa=D&source=editors&ust=1696873412606850&usg=AOvVaw0IF8bdtDWgRlAiW3tIoyXg)
for more details in IPC.

The user should allocate memory using `cuMemCreate` and again the
user has to explicitly provide `CU_MEM_LOCATION_TYPE_HOST_NUMA` as
the location type and *numaID* as the location identifier. In
addition `CU_MEM_HANDLE_TYPE_FABRIC` should be defined as the
requested handle type. The following code snippet shows allocating
physical memory on Node A:

```c++
CUmemAllocationProp prop{};
prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC;
prop.location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA;
prop.location.id = numaId;
size_t granularity = 0;
cuMemGetAllocationGranularity(&granularity, &prop,
                              MEM_ALLOC_GRANULARITY_MINIMUM);
size_t padded_size = ROUND_UP(size, granularity);
size_t page_size = ...;
assert(padded_size % page_size == 0);
CUmemGenericAllocationHandle allocHandle;
cuMemCreate(&allocHandle, padded_size, &prop, 0);
```

After creating allocation handle using `cuMemCreate` the user can
export that handle to the other node, Node B, calling
`cuMemExportToShareableHandle`:

```c++
cuMemExportToShareableHandle(&fabricHandle, allocHandle,
                             CU_MEM_HANDLE_TYPE_FABRIC, 0);
// At this point, fabricHandle should be sent to Node B via TCP/IP.
```

On Node B, the handle can be imported using
`cuMemImportFromShareableHandle` and treated as any other fabric
handle

```c++
// At this point, fabricHandle should be received from Node A via TCP/IP.
CUmemGenericAllocationHandle allocHandle;
cuMemImportFromShareableHandle(&allocHandle, &fabricHandle,
                               CU_MEM_HANDLE_TYPE_FABRIC);
```

When handle is imported at Node B, then the user can reserve an address
space and map it locally in a regular fashion:

```c++
size_t granularity = 0;
cuMemGetAllocationGranularity(&granularity, &prop,
                              MEM_ALLOC_GRANULARITY_MINIMUM);
size_t padded_size = ROUND_UP(size, granularity);
size_t page_size = ...;
assert(padded_size % page_size == 0);
CUdeviceptr dptr;
cuMemAddressReserve(&dptr, padded_size, 0, 0, 0);
cuMemMap(dptr, padded_size, 0, allocHandle, 0);
```

As the final step, the user should give appropriate accesses to each of
the local GPUs at Node B. An example code snippet that gives read and
write access to eight local GPUs:

```c++
// Give all 8 local  GPUS access to exported EGM memory located on Node A.                                                               |
CUmemAccessDesc accessDesc[8];
for (int i = 0; i < 8; i++) {
   accessDesc[i].location.type = CU_MEM_LOCATION_TYPE_DEVICE;
   accessDesc[i].location.id = i;
   accessDesc[i].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
}
cuMemSetAccess(dptr, size, accessDesc, 8);
```
