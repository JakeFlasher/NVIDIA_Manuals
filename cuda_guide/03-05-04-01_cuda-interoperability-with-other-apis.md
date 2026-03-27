---
title: "3.5.4.1. CUDA Interoperability with other APIs"
section: "3.5.4.1"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/feature-survey.html#cuda-interoperability-with-other-apis"
---

### [3.5.4.1. CUDA Interoperability with other APIs](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced#cuda-interoperability-with-other-apis)[](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/#cuda-interoperability-with-other-apis "Permalink to this headline")

There are other mechanisms than CUDA for running code on GPUs. The application GPUs were originally built to accelerate, computer graphics, uses its own set of APIs such as Direct3D and Vulkan. Applications may wish to use one of the graphics APIs for 3D rendering while performing computations in CUDA. CUDA provides mechanisms for exchanging data stored on the GPU between the CUDA contexts and the GPU contexts used by the 3D APIs. For example, an application may perform a simulation using CUDA, and then use a 3D API to create visualizations of the results. This is achieved by making some buffers readable and/or writeable from both CUDA and the graphics API.

The same mechanisms which allow sharing of buffers with graphics APIs are also used to share buffers with communications mechanisms which can enable rapid, direct GPU-to-GPU communication within multi-node environments.

[Section 4.19](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/graphics-interop.html#cuda-interoperability) describes how CUDA interoperates with other GPU APIs and how to share data between CUDA and other APIs, providing specific examples for a number of different APIs.
