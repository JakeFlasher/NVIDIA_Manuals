---
title: "2.4.4. Summary"
section: "2.4.4"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/understanding-memory.html#summary"
---

## [2.4.4. Summary](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics#summary)[](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/#summary "Permalink to this headline")

- On Linux platforms with heterogeneous memory management (HMM) or address translation services (ATS), all system-allocated memory is managed memory
- On Linux platforms without HMM or ATS, on Tegra processors, and on all Windows platforms, managed memory must be allocated using CUDA:

> - `cudaMallocManaged` or
> - `cudaMallocFromPoolAsync` with a pool created with `allocType=cudaMemAllocationTypeManaged`
> - Global variables with `__managed__` specifier
- On Windows and Tegra processors, unified memory has limitations
- On NVLINK C2C connected systems with ATS, device memory allocated with `cudaMalloc` can be directly accessed from the CPU or other GPUs
