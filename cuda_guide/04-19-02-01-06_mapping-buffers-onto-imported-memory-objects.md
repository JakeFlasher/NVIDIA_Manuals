---
title: "4.19.2.1.6. Mapping buffers onto imported memory objects"
section: "4.19.2.1.6"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/graphics-interop.html#mapping-buffers-onto-imported-memory-objects"
---

#### [4.19.2.1.6. Mapping buffers onto imported memory objects](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#mapping-buffers-onto-imported-memory-objects)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#mapping-buffers-onto-imported-memory-objects "Permalink to this headline")

After importing a memory object, they have to be mapped before they can be used.
A device pointer can be mapped onto an imported memory object as shown below. The offset and size of the mapping must match that specified when creating the mapping using the corresponding Vulkan API.
All mapped device pointers must be freed using `cudaFree()`.

```cuda
    // from the CUDA example `simpleVulkan`, continuation of function `importCudaExternalMemory`
    cudaExternalMemoryBufferDesc externalMemBufferDesc = {};
    externalMemBufferDesc.offset = 0;
    externalMemBufferDesc.size = size;
    externalMemBufferDesc.flags = 0;

    checkCudaErrors(cudaExternalMemoryGetMappedBuffer(cudaPtr, cudaMem,
                                                      &externalMemBufferDesc));
  }
```
