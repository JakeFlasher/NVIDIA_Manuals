---
title: "Convolution"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/profiler.html#convolution"
---

# [Convolution](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#convolution)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#convolution "Permalink to this headline")

The CUTLASS Profiler is capable of executing 2-D and 3-D convolution problems for forwards and backwards
operator variants.

The CUTLASS Profiler can be built with cuDNN enabled to use as a reference implementation. If CMake detects
the cuDNN library available in the system, it is included as a dependency. This may be explicitly overridden
with CMake flag `CUTLASS_ENABLE_CUDNN`.

```bash
$ cmake .. -DCUTLASS_LIBRARY_OPERATIONS=conv2d -DCUTLASS_ENABLE_CUDNN=OFF
...
$ make -j16 cutlass_profiler
```
