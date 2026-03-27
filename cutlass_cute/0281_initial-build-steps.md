---
title: "Initial build steps"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/quickstart.html#initial-build-steps"
---

## [Initial build steps](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#initial-build-steps)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#initial-build-steps "Permalink to this headline")

Construct a build directory and run CMake.

```bash
$ export CUDACXX=${CUDA_INSTALL_PATH}/bin/nvcc

$ mkdir build && cd build

$ cmake .. -DCUTLASS_NVCC_ARCHS=90a            # compiles for NVIDIA Hopper GPU architecture
$ cmake .. -DCUTLASS_NVCC_ARCHS=100a           # compiles for NVIDIA Blackwell SM100 GPU architecture
```

If your goal is strictly to build only the CUTLASS Profiler and to minimize compilation time, we suggest
executing the following CMake command in an empty `build/` directory.

```bash
$ cmake .. -DCUTLASS_NVCC_ARCHS=90a -DCUTLASS_ENABLE_TESTS=OFF -DCUTLASS_UNITY_BUILD_ENABLED=ON
```

This reduces overall compilation time by excluding unit tests and enabling the unity build.

You may reduce build times by compiling only certain operations by setting the `CUTLASS_LIBRARY_OPERATIONS` flag as shown below,
executed from an empty `build/` directory. This only compiles 2-D convolution kernels.

```bash
$ cmake .. -DCUTLASS_NVCC_ARCHS=90a -DCUTLASS_LIBRARY_OPERATIONS=conv2d
```

You may also filter kernels by name by supplying a filter string with flag `CUTLASS_LIBRARY_KERNELS`. For example the below command selects only CUTLASS-3 kernels.

```bash
$ cmake .. -DCUTLASS_NVCC_ARCHS=90a -DCUTLASS_LIBRARY_KERNELS=cutlass3x*
```

See more examples on selectively compiling CUTLASS GEMM and convolution kernels [here](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#example-cmake-commands).

You may explicitly exclude cuBLAS and cuDNN as dependencies with the following CMake flags.

- `-DCUTLASS_ENABLE_CUBLAS=OFF`
- `-DCUTLASS_ENABLE_CUDNN=OFF`
