---
title: "4.19.2.1.10. OpenGL Interoperability"
section: "4.19.2.1.10"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/graphics-interop.html#opengl-interoperability-ext-res-int"
---

#### [4.19.2.1.10. OpenGL Interoperability](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#opengl-interoperability-ext-res-int)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#opengl-interoperability-ext-res-int "Permalink to this headline")

Traditional OpenGL-CUDA interop as outlined in [OpenGL Interoperability](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#opengl-interoperability) works by CUDA directly consuming handles created in OpenGL. However, since OpenGL can also consume memory and synchronization objects created in Vulkan, there exists an alternative approach to doing OpenGL-CUDA interop.
Essentially, memory and synchronization objects exported by Vulkan could be imported into both, OpenGL and CUDA, and then used to coordinate memory accesses between OpenGL and CUDA. Please refer to the following OpenGL extensions for further details on how to import memory and synchronization objects exported by Vulkan:

- `GL_EXT_memory_object`
- `GL_EXT_memory_object_fd`
- `GL_EXT_memory_object_win32`
- `GL_EXT_semaphore`
- `GL_EXT_semaphore_fd`
- `GL_EXT_semaphore_win32`
