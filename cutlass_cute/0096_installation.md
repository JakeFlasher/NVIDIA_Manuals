---
title: "Installation"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/quick_start.html#installation"
---

## [Installation](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL#installation)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/#installation "Permalink to this headline")

Before installing the latest version, you need to uninstall any previous CUTLASS DSL Installation.

```bash
pip uninstall nvidia-cutlass-dsl nvidia-cutlass-dsl-libs-base nvidia-cutlass-dsl-libs-cu13 -y
```

To ensure compatibility with the examples and code on [GitHub](https://github.com/NVIDIA/cutlass/tree/main),
use the [setup.sh](https://github.com/NVIDIA/cutlass/blob/main/python/CuTeDSL/setup.sh) file from the corresponding commit in the repository.

```bash
git clone https://github.com/NVIDIA/cutlass.git

# For CUDA Toolkit 12.9:
./cutlass/python/CuTeDSL/setup.sh --cu12

# For CUDA Toolkit 13.1:
./cutlass/python/CuTeDSL/setup.sh --cu13
```

If you just want to try out the last known stable release of the CUTLASS DSL (may not be compatible with the latest examples and code), run:

```bash
# For CUDA Toolkit 12.9:
pip install nvidia-cutlass-dsl

# For CUDA Toolkit 13.1:
pip install nvidia-cutlass-dsl[cu13]
```

The `nvidia-cutlass-dsl` wheel includes everything needed to generate GPU kernels. It requires
the same NVIDIA driver version as the corresponding [CUDA Toolkit](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html)
(CUDA Toolkit 12.9 or CUDA Toolkit 13.1).
