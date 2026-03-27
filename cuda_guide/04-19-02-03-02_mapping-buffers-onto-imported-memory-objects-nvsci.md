---
title: "4.19.2.3.2. Mapping Buffers onto Imported Memory Objects"
section: "4.19.2.3.2"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/graphics-interop.html#mapping-buffers-onto-imported-memory-objects-nvsci"
---

#### [4.19.2.3.2. Mapping Buffers onto Imported Memory Objects](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#mapping-buffers-onto-imported-memory-objects-nvsci)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#mapping-buffers-onto-imported-memory-objects-nvsci "Permalink to this headline")

A device pointer can be mapped onto an imported memory object as shown below. The offset and size of the mapping can be filled as per the attributes of the allocated `NvSciBufObj`. All mapped device pointers must be freed using `cudaFree()`.

```c++
void * mapBufferOntoExternalMemory(cudaExternalMemory_t extMem, unsigned long long offset, unsigned long long size) {
    void *ptr = NULL;
    cudaExternalMemoryBufferDesc desc = {};

    memset(&desc, 0, sizeof(desc));

    desc.offset = offset;
    desc.size = size;

    cudaExternalMemoryGetMappedBuffer(&ptr, extMem, &desc);

    // Note: 'ptr' must eventually be freed using cudaFree()
    return ptr;
}
```
