---
title: "Building for Multiple Architectures"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/quickstart.html#building-for-multiple-architectures"
---

## [Building for Multiple Architectures](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#building-for-multiple-architectures)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#building-for-multiple-architectures "Permalink to this headline")

To minimize compilation time, specific GPU architectures can be enabled via the CMake command,
selected by [CUDA Compute Capability.](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities)

**NVIDIA Blackwell Architecture.**

```bash
$ cmake .. -DCUTLASS_NVCC_ARCHS=100a              # compiles for NVIDIA Blackwell GPU architecture
```

**NVIDIA Hopper Architecture.**

```bash
$ cmake .. -DCUTLASS_NVCC_ARCHS=90a              # compiles for NVIDIA Hopper GPU architecture
```

**NVIDIA Ampere Architecture.**

```bash
$ cmake .. -DCUTLASS_NVCC_ARCHS=80               # compiles for NVIDIA Ampere GPU architecture
```

**NVIDIA Turing Architecture.**

```bash
$ cmake .. -DCUTLASS_NVCC_ARCHS=75               # compiles for NVIDIA Turing GPU architecture
```

**NVIDIA Volta Architecture.**

```bash
$ cmake .. -DCUTLASS_NVCC_ARCHS=70               # compiles for NVIDIA Volta GPU architecture
```

**NVIDIA Pascal Architecture.**

```bash
$ cmake .. -DCUTLASS_NVCC_ARCHS="60;61"          # compiles for NVIDIA Pascal GPU architecture
```

**NVIDIA Maxwell Architecture.**

```bash
$ cmake .. -DCUTLASS_NVCC_ARCHS="50;53"          # compiles for NVIDIA Maxwell GPU architecture
```
