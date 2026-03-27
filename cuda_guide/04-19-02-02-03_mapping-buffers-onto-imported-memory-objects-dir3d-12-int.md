---
title: "4.19.2.2.3. Mapping Buffers onto Imported Memory Objects"
section: "4.19.2.2.3"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/graphics-interop.html#mapping-buffers-onto-imported-memory-objects-dir3d-12-int"
---

#### [4.19.2.2.3. Mapping Buffers onto Imported Memory Objects](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#mapping-buffers-onto-imported-memory-objects-dir3d-12-int)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#mapping-buffers-onto-imported-memory-objects-dir3d-12-int "Permalink to this headline")

A device pointer can be mapped onto an imported memory object as shown below.
The offset and size of the mapping must match that specified when creating the mapping using the corresponding Direct3D12 API.
All mapped device pointers must be freed using `cudaFree()`.

```cuda
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
