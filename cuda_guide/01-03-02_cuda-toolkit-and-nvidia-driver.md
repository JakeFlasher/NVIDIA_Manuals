---
title: "1.3.2. CUDA Toolkit and NVIDIA Driver"
section: "1.3.2"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/cuda-platform.html#cuda-toolkit-and-nvidia-driver"
---

## [1.3.2. CUDA Toolkit and NVIDIA Driver](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction#cuda-toolkit-and-nvidia-driver)[](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/#cuda-toolkit-and-nvidia-driver "Permalink to this headline")

The _NVIDIA Driver_ can be thought of as the operating system of the GPU. The NVIDIA Driver is a software component which must be installed on the host system’s operating system and is necessary for all GPU uses, including display and graphical functionality. The NVIDIA Driver is foundational to the CUDA platform. In addition to CUDA, the NVIDIA Driver provides all other methods of using the GPU, for example Vulkan and Direct3D. The NVIDIA Driver has version numbers such as r580.

The _CUDA Toolkit_ is a set of libraries, headers, and tools for writing, building, and analyzing software which utilizes GPU computing. The CUDA Toolkit is a separate software product from the NVIDIA driver

The _CUDA runtime_ is a special case of one of the libraries provided by the CUDA Toolkit. The CUDA runtime provides both an API and some language extensions to handle common tasks such as allocating memory, copying data between GPUs and other GPUs or CPUs, and launching kernels. The API components of the CUDA runtime are referred to as the CUDA runtime API.

The [CUDA Compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/index.html) document provides full details of compatibility between different GPUs, NVIDIA Drivers, and CUDA Toolkit versions.
