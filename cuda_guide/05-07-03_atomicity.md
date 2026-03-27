---
title: "5.7.3. Atomicity"
section: "5.7.3"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cuda-cpp-memory-model.html#atomicity"
---

## [5.7.3. Atomicity](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#atomicity)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#atomicity "Permalink to this headline")

An atomic operation is atomic at the scope it specifies if:

- it specifies a scope other than `cuda::thread_scope_system`, **or**
- the scope is `cuda::thread_scope_system` **and**:
  - it affects an object in [system allocated memory](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/unified-memory.html#um-details-intro) and [pageableMemoryAccess](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg49e2f8c2c0bd6fe264f2fc970912e5cddc80992427a92713e699953a6d249d6f) is `1` [0],  **or**
  - it affects an object in [managed memory](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/unified-memory.html#um-details-intro) and [concurrentManagedAccess](https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html#structcudaDeviceProp_116f9619ccc85e93bc456b8c69c80e78b) is `1`, **or**
  - it affects an object in [mapped memory](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/understanding-memory.html#memory-mapped-memory) and [hostNativeAtomicSupported](https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html#structcudaDeviceProp_1ef82fd7d1d0413c7d6f33287e5b6306f) is `1`, **or**
  - it is a load or store that affects a naturally-aligned object of
sizes `1`, `2`, `4`, `8`, or `16` bytes on [mapped memory](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/understanding-memory.html#memory-mapped-memory) [1], **or**
  - it affects an object in GPU memory, only GPU threads access it, **and**
    - [cudaDeviceGetP2PAttribute](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g2f597e2acceab33f60bd61c41fea0c1b) `(&val, cudaDevP2PAttrNativeAtomicSupported, srcDev, dstDev)`
between each accessing `srcDev` and the GPU where the object resides, `dstDev`, is `1`, **or**
    - only GPU threads from a single GPU concurrently access it.

> **Note**
>
> - [0] If [PageableMemoryAccessUsesHostPagetables](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg49e2f8c2c0bd6fe264f2fc970912e5cdc228cf8983c97d0e035da72a71494eaa) is `0` then atomic operations to memory mapped file or `hugetlbfs` allocations are not atomic.
> - [1] If [hostNativeAtomicSupported](https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html#structcudaDeviceProp_1ef82fd7d1d0413c7d6f33287e5b6306f) is `0`, atomic load or store operations at system scope that affect a
> naturally-aligned 16-byte wide object in [system allocated memory](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/unified-memory.html#um-details-intro) or [mapped memory](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/understanding-memory.html#memory-mapped-memory) require system
> support. NVIDIA is not aware of any system that lacks this support and there is no CUDA API query available to
> detect such systems.

For more information on [system allocated memory](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/unified-memory.html#um-details-intro), [managed memory](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/unified-memory.html#um-details-intro), [mapped memory](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/understanding-memory.html#memory-mapped-memory),
CPU memory, and GPU memory, see the relevant sections of this guide.
