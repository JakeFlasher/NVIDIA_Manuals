---
title: "2.5.3.2. Host Code Compilation Notes"
section: "2.5.3.2"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/nvcc.html#host-code-compilation-notes"
---

### [2.5.3.2. Host Code Compilation Notes](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics#host-code-compilation-notes)[](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/#host-code-compilation-notes "Permalink to this headline")

Compilation units, namely a source file and its headers, that do not contain device code or symbols can be compiled directly with a host compiler. If any compilation unit uses CUDA runtime API functions, the application must be linked with the CUDA runtime library. The CUDA runtime is available as both a static and a shared library, `libcudart_static` and `libcudart`, respectively. By default, `nvcc` links against the static CUDA runtime library. To use the shared library version of the CUDA runtime, pass the flag `--cudart=shared` to `nvcc` on the compile or link command.

`nvcc` allows the host compiler used for host functions to be specified via the `-ccbin <compiler>` argument. The environment variable `NVCC_CCBIN` can also be defined to specify the host compiler used by `nvcc`. The `-Xcompiler` argument to `nvcc` passes through arguments to the host compiler. For example, in the example below, the `-O3` argument is passed to the host compiler by `nvcc`.

```bash
nvcc example.cu -ccbin=clang++

export NVCC_CCBIN='gcc'
nvcc example.cu -Xcompiler=-O3
```
