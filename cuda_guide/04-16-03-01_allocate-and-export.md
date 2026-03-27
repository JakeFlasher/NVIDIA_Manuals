---
title: "4.16.3.1. Allocate and Export"
section: "4.16.3.1"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/virtual-memory-management.html#allocate-and-export"
---

### [4.16.3.1. Allocate and Export](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#allocate-and-export)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#allocate-and-export "Permalink to this headline")

**Allocating Physical Memory**
The first step in memory allocation using virtual memory management APIs is to
create a physical memory chunk that will provide a backing for the allocation.
In order to allocate physical memory, applications must use the
`cuMemCreate` API. The allocation created by this function does not have any
device or host mappings. The function argument
`CUmemGenericAllocationHandle` describes the properties of the memory to
allocate such as the location of the allocation, if the allocation is going to
be shared to another process (or graphics APIs), or the physical
attributes of the memory to be allocated. Users must ensure the requested
allocation’s size is aligned to appropriate granularity. Information
regarding an allocation’s granularity requirements can be queried using
`cuMemGetAllocationGranularity`.

**OS-Specific Handle (Linux)**

```cuda
CUmemGenericAllocationHandle allocatePhysicalMemory(int device, size_t size) {
    CUmemAllocationHandleType handleType = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = device;
    prop.requestedHandleType = handleType;

    size_t granularity = 0;
    cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);

    // Ensure size matches granularity requirements for the allocation
    size_t padded_size = ROUND_UP(size, granularity);

    // Allocate physical memory
    CUmemGenericAllocationHandle allocHandle;
    cuMemCreate(&allocHandle, padded_size, &prop, 0);

    return allocHandle;
}
```

**Fabric Handle**

```cuda
CUmemGenericAllocationHandle allocatePhysicalMemory(int device, size_t size) {
    CUmemAllocationHandleType handleType = CU_MEM_HANDLE_TYPE_FABRIC;
    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = device;
    prop.requestedHandleType = handleType;

    size_t granularity = 0;
    cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);

    // Ensure size matches granularity requirements for the allocation
    size_t padded_size = ROUND_UP(size, granularity);

    // Allocate physical memory
    CUmemGenericAllocationHandle allocHandle;
    cuMemCreate(&allocHandle, padded_size, &prop, 0);

    return allocHandle;
}
```

> **Note**
>
> The memory allocated by `cuMemCreate` is referenced by the
> `CUmemGenericAllocationHandle` it returns. Note that this reference is
> not a pointer and its memory is not accessible yet.

> **Note**
>
> Properties of the allocation handle can be queried using
> `cuMemGetAllocationPropertiesFromHandle`.

**Exporting Memory Handle**
The CUDA virtual memory management API expose a new mechanism for interprocess
communication using handles to exchange necessary information about the
allocation and physical address space. One can export handles for OS-specific
IPC or fabric-specific IPC. OS-specific IPC handles can only be used on a
single-node setup. Fabric-specific handles can be used on a single or multi
node setups.

**OS-Specific Handle (Linux)**

```cuda
CUmemAllocationHandleType handleType = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
CUmemGenericAllocationHandle handle = allocatePhysicalMemory(0, 1<<21);
int fd;
cuMemExportToShareableHandle(&fd, handle, handleType, 0);
```

**Fabric Handle**

```cuda
CUmemAllocationHandleType handleType = CU_MEM_HANDLE_TYPE_FABRIC;
CUmemGenericAllocationHandle handle = allocatePhysicalMemory(0, 1<<21);
CUmemFabricHandle fh;
cuMemExportToShareableHandle(&fh, handle, handleType, 0);
```

> **Note**
>
> OS-specific handles require all processes to be part of the same OS.

> **Note**
>
> Fabric-specific handles require IMEX channels to be enabled by sysadmin.

The [memMapIpcDrv](https://github.com/NVIDIA/cuda-samples/tree/master/Samples/3_CUDA_Features/memMapIPCDrv/)
sample can be used as an example for using IPC with VMM allocations.
