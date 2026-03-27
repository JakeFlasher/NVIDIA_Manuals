---
title: "1.1.1. The Graphics Processing Unit"
section: "1.1.1"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/introduction.html#the-graphics-processing-unit"
---

## [1.1.1. The Graphics Processing Unit](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction#the-graphics-processing-unit)[](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/#the-graphics-processing-unit "Permalink to this headline")

Born as a special-purpose processor for 3D graphics, the _Graphics Processing Unit_ (GPU) started out as fixed-function hardware to accelerate parallel operations in real-time 3D rendering. Over successive generations, GPUs became more programmable. By 2003, some stages of the graphics pipeline became fully programmable, running custom code in parallel for each component of a 3D scene or an image.

In 2006, NVIDIA introduced the _Compute Unified Device Architecture_ (CUDA) to enable any computational workload to use the throughput capability of GPUs independent of graphics APIs.

Since then, CUDA and GPU computing have been used to accelerate computational workloads of nearly every type, from scientific simulations such as fluid dynamics or energy transport to business applications like databases and analytics. Moreover, the capability and programmability of GPUs has been foundational to the advancement of new algorithms and technologies ranging from image classification to generative artificial intelligence such as diffusion or large language models.
