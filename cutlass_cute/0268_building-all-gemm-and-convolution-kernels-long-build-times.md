---
title: "Building all GEMM and Convolution kernels (long build times)"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/overview.html#building-all-gemm-and-convolution-kernels-long-build-times"
---

## [Building all GEMM and Convolution kernels (long build times)](https://docs.nvidia.com/cutlass/latest#building-all-gemm-and-convolution-kernels-long-build-times)[](https://docs.nvidia.com/cutlass/latest/#building-all-gemm-and-convolution-kernels-long-build-times "Permalink to this headline")

By default, only one tile size is instantiated for each data type, math instruction, and layout.
To instantiate all, set the following environment variable when running CMake from an empty `build/` directory.
Beware, this results in _tens of thousands_ of kernels and long build times.
This would also result in a large binary size and on some platforms linker to fail on building the library.
Therefore, it’s highly recommended to generate only a subset of kernels as demonstrated in the sub-section below.

```bash
$ cmake .. -DCUTLASS_NVCC_ARCHS=90a -DCUTLASS_LIBRARY_KERNELS=all
...
$ make cutlass_profiler -j16
```
