---
title: "4.17.1.1. EGM Platforms: System topology"
section: "4.17.1.1"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/extended-gpu-memory.html#egm-platforms-system-topology"
---

### [4.17.1.1. EGM Platforms: System topology](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#egm-platforms-system-topology)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#egm-platforms-system-topology "Permalink to this headline")

Currently, EGM can be enabled in several platforms: **(1) Single-Node, Single-GPU**:
Consists of an Arm-based CPU, CPU attached memory, and a GPU. Between the CPU
and the GPU there is a high bandwidth C2C (Chip-to-Chip) interconnect.
**(2) Single-Node, Multi-GPU**: Consists of ARM-based CPUs, each with attached memory,
and multiple GPUs connected through an NVLink-based network.
**(3) Multi-Node, Multi-GPU**: Two or more single-node systems, each as in (1) or (2) above,
connected through an NVLink-based network.

> **Note**
>
> Using `cgroups` to limit available devices will block routing over EGM
> and cause performance issues. Use `CUDA_VISIBLE_DEVICES` instead.
