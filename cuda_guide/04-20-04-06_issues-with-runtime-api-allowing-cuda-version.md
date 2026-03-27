---
title: "4.20.4.6. Issues with Runtime API allowing CUDA Version"
section: "4.20.4.6"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/driver-entry-point-access.html#issues-with-runtime-api-allowing-cuda-version"
---

### [4.20.4.6. Issues with Runtime API allowing CUDA Version](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#issues-with-runtime-api-allowing-cuda-version)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#issues-with-runtime-api-allowing-cuda-version "Permalink to this headline")

Unless specified otherwise, the CUDA runtime API `cudaGetDriverEntryPointByVersion` will have similar implications as the driver entry point `cuGetProcAddress` since it allows for the user to request a specific CUDA driver version.
