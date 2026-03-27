---
title: "2.5.1. CUDA Source Files and Headers"
section: "2.5.1"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/nvcc.html#cuda-source-files-and-headers"
---

## [2.5.1. CUDA Source Files and Headers](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics#cuda-source-files-and-headers)[](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/#cuda-source-files-and-headers "Permalink to this headline")

Source files compiled with `nvcc` may contain a combination of host code, which executes on the CPU, and device code that executes on the GPU. `nvcc` accepts the common C/C++ source file extensions `.c`, `.cpp`, `.cc`, `.cxx` for host-only code and `.cu` for files that contain device code or a mix of host and device code. Headers containing device code typically adopt the `.cuh` extension to distinguish them from host-only code headers `.h`, `.hpp`, `.hh`, `.hxx`, etc.

| File Extension | Description | Content |
| --- | --- | --- |
| `.c` | C source file | Host-only code |
| `.cpp`, `.cc`, `.cxx` | C++ source file | Host-only code |
| `.h`, `.hpp`, `.hh`, `.hxx` | C/C++ header file | Device code, host code, mix of host/device code |
| `.cu` | CUDA source file | Device code, host code, mix of host/device code |
| `.cuh` | CUDA header file | Device code, host code, mix of host/device code |
