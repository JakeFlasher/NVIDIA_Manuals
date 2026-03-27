---
title: "CUTLASS Profiler"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/profiler.html#cutlass-profiler"
---

# [CUTLASS Profiler](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#cutlass-profiler)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#cutlass-profiler "Permalink to this headline")

The CUTLASS Profiler is a command-line driven test and profiling environment for CUTLASS computations
defined in the CUTLASS Instance Library. The CUTLASS Profiler is capable of executing each GEMM, Sparse Gemm,
Conv2d, and Conv3d kernel.

The CUTLASS Profiler may be compiled with:

```bash
$ make cutlass_profiler -j
```

To limit compilation time, only one tile size (typically 128x128) and threadblock cluster size (typically 2x1x1) is instantiated for each data type,
math instruction, and layout. To instantiate all sizes, set the following environment variable when running CMake from an
empty `build/` directory.

```bash
$ cmake .. -DCUTLASS_NVCC_ARCHS="70;75;80" -DCUTLASS_LIBRARY_KERNELS=all  -DCUTLASS_UNITY_BUILD_ENABLED=ON
...
$ make cutlass_profiler -j
```

Enabling the unity build places multiple kernel instances in one compilation unit, thereby reducing size of the compiled
binary and avoiding linker limitations on some platforms.

The CUTLASS Profiler sources are stored in:

```bash
tools/
  profiler/
```
