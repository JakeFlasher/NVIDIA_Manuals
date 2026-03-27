---
title: "GEMM CMake Examples"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/quickstart.html#gemm-cmake-examples"
---

## [GEMM CMake Examples](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#gemm-cmake-examples)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#gemm-cmake-examples "Permalink to this headline")

**Example.** All GEMM kernels targeting NVIDIA Ampere Tensor Cores.

```bash
$ cmake .. -DCUTLASS_NVCC_ARCHS=80 -DCUTLASS_LIBRARY_KERNELS=tensorop*gemm
```

**Example.** All GEMM kernels targeting NVIDIA Turing Tensor Cores.

```bash
$ cmake .. -DCUTLASS_NVCC_ARCHS=75 -DCUTLASS_LIBRARY_KERNELS=tensorop*gemm
```

**Example.** All GEMM kernels with FP32 accumulation targeting NVIDIA Ampere, Turing, and Volta architectures.

```bash
$ cmake .. -DCUTLASS_NVCC_ARCHS="70;75;80" -DCUTLASS_LIBRARY_KERNELS=s*gemm
```

**Example.** All kernels which expect A and B to be column-major or row-major targeting NVIDIA Ampere, Turing, and Volta architectures.

```bash
$ cmake .. -DCUTLASS_NVCC_ARCHS="70;75;80" -DCUTLASS_LIBRARY_KERNELS=gemm*nn,gemm*tt
```

**Example.** All planar complex GEMM variants targeting NVIDIA Ampere, Turing, and Volta architectures.

```bash
$ cmake .. -DCUTLASS_NVCC_ARCHS="70;75;80" -DCUTLASS_LIBRARY_KERNELS=planar_complex
```
