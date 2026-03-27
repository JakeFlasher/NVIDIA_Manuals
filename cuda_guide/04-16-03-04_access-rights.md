---
title: "4.16.3.4. Access Rights"
section: "4.16.3.4"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/virtual-memory-management.html#access-rights"
---

### [4.16.3.4. Access Rights](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#access-rights)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#access-rights "Permalink to this headline")

CUDA’s virtual memory management APIs enable applications to explicitly protect
their VA ranges with access control mechanisms. Mapping the allocation to a
region of the address range using `cuMemMap` does not make the address
accessible, and would result in a program crash if accessed by a CUDA kernel.
Users must specifically select access control using the `cuMemSetAccess`
function on source and accessing devices. This allows or restricts access for
specific devices to a mapped  address range. The following code snippet
illustrates the usage for the function:

```c++
void setAccessOnDevice(int device, CUdeviceptr ptr, size_t size) {
    CUmemAccessDesc accessDesc = {};
    accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    accessDesc.location.id = device;
    accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

    // Make the address accessible
    cuMemSetAccess(ptr, size, &accessDesc, 1);
}
```

The access control mechanism exposed with VMM allows
users to be explicit about which allocations they want to share with other
peer devices on the system. As specified earlier, `cudaEnablePeerAccess`
forces all prior and future allocations made with `cudaMalloc` to be mapped to the
target peer device. This can be convenient in many cases as user doesn’t have
to worry about tracking the mapping state of every allocation to every device
in the system. But
this approach [has performance implications](https://devblogs.nvidia.com/introducing-low-level-gpu-virtual-memory-management/).
With access control at allocation granularity, VMM
allows peer mappings with minimal overhead.

The `vectorAddMMAP` [sample](https://github.com/NVIDIA/cuda-samples/tree/master/Samples/0_Introduction/vectorAddMMAP)
can be used as an example for using the Virtual
Memory Management APIs.
