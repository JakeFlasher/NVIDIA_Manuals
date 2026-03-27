---
title: "4.16.5.2. Compressible Memory"
section: "4.16.5.2"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/virtual-memory-management.html#compressible-memory"
---

### [4.16.5.2. Compressible Memory](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#compressible-memory)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#compressible-memory "Permalink to this headline")

Compressible memory can be used to accelerate accesses to data with
unstructured sparsity and other compressible data patterns. Compression can
save DRAM bandwidth, L2 read bandwidth, and L2 capacity depending on the data.
Applications that want to allocate compressible memory on
devices that support compute data compression can do so by setting
`CUmemAllocationProp::allocFlags::compressionType` to
`CU_MEM_ALLOCATION_COMP_GENERIC`. Users must query if device supports
Compute Data Compression by using
`CU_DEVICE_ATTRIBUTE_GENERIC_COMPRESSION_SUPPORTED`. The following code
snippet illustrates querying compressible memory support
`cuDeviceGetAttribute`.

```c++
int compressionSupported = 0;
cuDeviceGetAttribute(&compressionSupported, CU_DEVICE_ATTRIBUTE_GENERIC_COMPRESSION_SUPPORTED, device);
```

On devices that support compute data compression, users must opt in at
allocation time as shown below:

```c++
prop.allocFlags.compressionType = CU_MEM_ALLOCATION_COMP_GENERIC;
```

For a variety of reasons such as limited hardware resources, the allocation may not
have compression attributes. To verify that the flags worked, the user query the properties
of the allocated memory using `cuMemGetAllocationPropertiesFromHandle`.

```c++
CUmemAllocationProp allocationProp = {};
cuMemGetAllocationPropertiesFromHandle(&allocationProp, allocationHandle);

if (allocationProp.allocFlags.compressionType == CU_MEM_ALLOCATION_COMP_GENERIC)
{
    // Obtained compressible memory allocation
}
```
