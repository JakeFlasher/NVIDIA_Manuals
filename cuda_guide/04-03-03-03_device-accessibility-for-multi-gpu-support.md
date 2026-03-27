---
title: "4.3.3.3. Device Accessibility for Multi-GPU Support"
section: "4.3.3.3"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/stream-ordered-memory-allocation.html#device-accessibility-for-multi-gpu-support"
---

### [4.3.3.3. Device Accessibility for Multi-GPU Support](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#device-accessibility-for-multi-gpu-support)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#device-accessibility-for-multi-gpu-support "Permalink to this headline")

Like allocation accessibility controlled through the virtual memory
management APIs, memory pool allocation accessibility does not follow
`cudaDeviceEnablePeerAccess` or `cuCtxEnablePeerAccess`. For memory pools, the API
`cudaMemPoolSetAccess` modifies what devices can access allocations from a
pool. By default, allocations are accessible only from the device where the
allocations are located. This access cannot be revoked. To enable access from
other devices, the accessing device must be peer capable with the memory
pool’s device. This can be verified with `cudaDeviceCanAccessPeer`. If the peer capability
is not checked, the set access may fail with `cudaErrorInvalidDevice`. However, if no
allocations had been made from the pool, the `cudaMemPoolSetAccess` call may
succeed even when the devices are not peer capable. In this case, the next
allocation from the pool will fail.

It is worth noting that `cudaMemPoolSetAccess` affects all allocations from
the memory pool, not just future ones. Likewise, the accessibility reported by
`cudaMemPoolGetAccess` applies to all allocations from the pool, not just
future ones. Changing the accessibility settings of a pool for a
given GPU frequently is not recommended. That is, once a pool is made accessible from a
given GPU, it should remain accessible from that GPU for the lifetime of the
pool.

```c++
// snippet showing usage of cudaMemPoolSetAccess:
cudaError_t setAccessOnDevice(cudaMemPool_t memPool, int residentDevice,
              int accessingDevice) {
    cudaMemAccessDesc accessDesc = {};
    accessDesc.location.type = cudaMemLocationTypeDevice;
    accessDesc.location.id = accessingDevice;
    accessDesc.flags = cudaMemAccessFlagsProtReadWrite;

    int canAccess = 0;
    cudaError_t error = cudaDeviceCanAccessPeer(&canAccess, accessingDevice,
              residentDevice);
    if (error != cudaSuccess) {
        return error;
    } else if (canAccess == 0) {
        return cudaErrorPeerAccessUnsupported;
    }

    // Make the address accessible
    return cudaMemPoolSetAccess(memPool, &accessDesc, 1);
}
```
