---
title: "4.17.2.2.2. Using CUDA Memory Pool"
section: "4.17.2.2.2"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/extended-gpu-memory.html#using-cuda-memory-pool"
---

#### [4.17.2.2.2. Using CUDA Memory Pool](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#using-cuda-memory-pool)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#using-cuda-memory-pool "Permalink to this headline")

To define EGM, the user can create a memory pool on a node and give
access to peers. In this case, the user has to explicitly define
`cudaMemLocationTypeHostNuma` as the location type and *numaId*
as the location identifier. The following code snippet shows creating a
memory pool `cudaMemPoolCreate`:

```c++
cudaSetDevice(homeDevice);
cudaMemPoolProps props{};
props.allocType = cudaMemAllocationTypePinned;
props.location.type = cudaMemLocationTypeHostNuma;
props.location.id = numaId;
cudaMemPoolCreate(&memPool, &props);
```

Additionally, for direct connect peer access, it is also possible to use
the existing peer access API, `cudaMemPoolSetAccess`. An example
for an *accessingDevice* is shown in the following code snippet:

```c++
cudaMemAccessDesc desc{};
desc.flags = cudaMemAccessFlagsProtReadWrite;
desc.location.type = cudaMemLocationTypeDevice;
desc.location.id = accessingDevice;
cudaMemPoolSetAccess(memPool, &desc, 1);
```

When the memory pool is created, and accesses are given, the user can
set created memory pool to the *residentDevice* and start allocating
memory using `cudaMallocAsync`:

```c++
cudaDeviceSetMemPool(residentDevice, memPool);
cudaMallocAsync(&ptr, size, memPool, stream);
```

> **Note**
>
> EGM is mapped with 2MB pages. Therefore, users may encounter more TLB
> misses when accessing very large allocations.
