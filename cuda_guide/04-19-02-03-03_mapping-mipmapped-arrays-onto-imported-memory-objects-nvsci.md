---
title: "4.19.2.3.3. Mapping Mipmapped Arrays onto Imported Memory Objects"
section: "4.19.2.3.3"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/graphics-interop.html#mapping-mipmapped-arrays-onto-imported-memory-objects-nvsci"
---

#### [4.19.2.3.3. Mapping Mipmapped Arrays onto Imported Memory Objects](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#mapping-mipmapped-arrays-onto-imported-memory-objects-nvsci)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#mapping-mipmapped-arrays-onto-imported-memory-objects-nvsci "Permalink to this headline")

A CUDA mipmapped array can be mapped onto an imported memory object as shown below. The offset, dimensions and format can be filled as per the attributes of the allocated `NvSciBufObj`. All mapped mipmapped arrays must be freed using `cudaFreeMipmappedArray()`. The following code sample shows how to convert NvSciBuf attributes into the corresponding CUDA parameters when mapping mipmapped arrays onto imported memory objects.

> **Note**
>
> The number of mip levels must be 1.

```c++
cudaMipmappedArray_t mapMipmappedArrayOntoExternalMemory(cudaExternalMemory_t extMem, unsigned long long offset, cudaChannelFormatDesc *formatDesc, cudaExtent *extent, unsigned int flags, unsigned int numLevels) {
    cudaMipmappedArray_t mipmap = NULL;
    cudaExternalMemoryMipmappedArrayDesc desc = {};

    memset(&desc, 0, sizeof(desc));

    desc.offset = offset;
    desc.formatDesc = *formatDesc;
    desc.extent = *extent;
    desc.flags = flags;
    desc.numLevels = numLevels;

    // Note: 'mipmap' must eventually be freed using cudaFreeMipmappedArray()
    cudaExternalMemoryGetMappedMipmappedArray(&mipmap, extMem, &desc);

    return mipmap;
}
```
