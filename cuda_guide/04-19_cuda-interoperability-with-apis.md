---
title: "4.19. CUDA Interoperability with APIs"
section: "4.19"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/graphics-interop.html#cuda-interoperability-with-apis"
---

# [4.19. CUDA Interoperability with APIs](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#cuda-interoperability-with-apis)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#cuda-interoperability-with-apis "Permalink to this headline")

Directly accessing GPU data from APIs in CUDA allows to read and write the data with CUDA kernels and thereby offering CUDA features while consuming them from other APIs.
There are two main concepts: the direct approach, [Graphics Interoperability](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#graphics-interoperability) with openGL and Direct3D[9-11] which enables to map the resources from the OpenGL and the Direct3D to the CUDA address space; and the more flexible [External resource interoperability](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#external-resource-interoperability),
where memory and synchronization objects can be accessed by importing and exporting through OS-level handles.
This is supported for the following APIs, Direct3D[11-12], Vulkan and the NVIDIA Software Communication Interface Interoperability.
