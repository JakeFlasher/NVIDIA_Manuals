---
title: "3.4.1. Multi-Device Context and Execution Management"
section: "3.4.1"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/multi-gpu-systems.html#multi-device-context-and-execution-management"
---

## [3.4.1. Multi-Device Context and Execution Management](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced#multi-device-context-and-execution-management)[](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/#multi-device-context-and-execution-management "Permalink to this headline")

The first steps that are required to for an application to use multiple GPUs
are to enumerate the available GPU devices,
select among the available devices as appropriate based on their
hardware properties, CPU affinity, and connectivity to peers,
and to create CUDA contexts for each device that the application will use.
