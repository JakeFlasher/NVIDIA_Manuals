---
title: "4.19.1. Graphics Interoperability"
section: "4.19.1"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/graphics-interop.html#graphics-interoperability"
---

## [4.19.1. Graphics Interoperability](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#graphics-interoperability)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#graphics-interoperability "Permalink to this headline")

Before accessing a Direct3D or an openGL resource, for example a VBO (vertex buffer object) with CUDA, it must be registered and mapped.
The registering with the according CUDA functions, see examples below, returns a CUDA graphics resource of type `struct cudaGraphicsResource`, which holds a CUDA device pointer or array.
To access the device data in a kernel, the resource must be mapped.  While the resource is registered, it can be mapped and unmapped as many times as necessary.
A mapped resource is accessed by kernels using the device memory address returned by `cudaGraphicsResourceGetMappedPointer()` for buffers and `cudaGraphicsSubResourceGetMappedArray()` for CUDA arrays.
Once the resource is no longer needed by CUDA, it can be unregistered.
These are the main steps:
1. Register the graphics buffer with CUDA
2. Map the resource
3. Access the device pointer or array of the mapped resource
4. Use device pointer or array in a CUDA kernel
4. Unmap the resource
5. Unregister the resource

Note that, registering a resource is costly and therefore ideally only called once per resource, however for each CUDA context which intends to use the resource, it is required to register the resource separately.
`cudaGraphicsResourceSetMapFlags()` can be called to specify usage hints (write-only, read-only) that the CUDA driver can use to optimize resource management.
Further note, that when accessing a resource through OpenGL, Direct3D, or a different CUDA context while it is mapped, it produces undefined results.
