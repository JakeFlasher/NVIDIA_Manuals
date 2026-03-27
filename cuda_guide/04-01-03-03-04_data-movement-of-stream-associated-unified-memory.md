---
title: "4.1.3.3.4. Data Movement of Stream Associated Unified Memory"
section: "4.1.3.3.4"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/unified-memory.html#data-movement-of-stream-associated-unified-memory"
---

#### [4.1.3.3.4. Data Movement of Stream Associated Unified Memory](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#data-movement-of-stream-associated-unified-memory)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#data-movement-of-stream-associated-unified-memory "Permalink to this headline")

Memcpy()/Memset() with stream associated unified memory behaves different on devices where `concurrentManagedAccess` is not set, the following rules apply:

If `cudaMemcpyHostTo*` is specified and the source data is unified memory, then it will be accessed from the host if it is coherently accessible from the host in the copy stream [(1)](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#um-legacy-memcpy-cit1); otherwise it will be accessed from the device. Similar rules apply to the destination when `cudaMemcpy*ToHost` is specified and the destination is unified memory.

If `cudaMemcpyDeviceTo*` is specified and the source data is unified memory, then it will be accessed from the device. The source must be coherently accessible from the device in the copy stream [(2)](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#um-legacy-memcpy-cit2); otherwise, an error is returned. Similar rules apply to the destination when `cudaMemcpy*ToDevice` is specified and the destination is unified memory.

If `cudaMemcpyDefault` is specified, then unified memory will be accessed from the host either if it cannot be coherently accessed from the device in the copy stream [(2)](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#um-legacy-memcpy-cit2) or if the preferred location for the data is `cudaCpuDeviceId` and it can be coherently accessed from the host in the copy stream [(1)](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#um-legacy-memcpy-cit1); otherwise, it will be accessed from the device.

When using `cudaMemset*()` with unified memory, the data must be coherently accessible from the device in the stream being used for the `cudaMemset()` operation [(2)](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#um-legacy-memcpy-cit2); otherwise, an error is returned.

When data is accessed from the device either by `cudaMemcpy*` or `cudaMemset*`, the stream of operation is considered to be active on the GPU. During this time, any CPU access of data that is associated with that stream or data that has global visibility, will result in a segmentation fault if the GPU has a zero value for the device attribute `concurrentManagedAccess`. The program must synchronize appropriately to ensure the operation has completed before accessing any associated data from the CPU.

> 1. Coherently accessible from the host in a given stream means that the memory neither has global visibility nor is it associated with the given stream.

> 2. Coherently accessible from the device in a given stream means that the memory either has global visibility or is associated with the given stream.
