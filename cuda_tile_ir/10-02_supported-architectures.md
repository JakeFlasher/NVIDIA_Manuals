---
title: "10.2. Supported Architectures"
section: "10.2"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/stability.html#supported-architectures"
---

## [10.2. Supported Architectures](https://docs.nvidia.com/cuda/tile-ir/latest/sections#supported-architectures)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#supported-architectures "Permalink to this headline")

**Tile IR** bytecode programs are portable across all supported architectures. A single bytecode
file can be compiled to any supported target or JIT-compiled by the driver at load time.

For ahead-of-time compilation, the target architecture is specified using the `--gpu-name` flag
with a supported NVDIA GPU architecture compute capability (CC) number
(e.g., `tileiras --gpu-name sm_80`). For JIT compilation, the driver automatically selects the
architecture of the target device. The following table lists the architectures supported by
**Tile IR** and the corresponding `--gpu-name` values.

| Family | Compute Capability | Example GPUs | Since |
| --- | --- | --- | --- |
| Ampere | sm_80 | A100, A30 | **Tile IR** 13.2 |
| Ampere | sm_86 | A40, RTX 3090 | **Tile IR** 13.2 |
| Ampere | sm_87, sm_88 | Jetson Orin | **Tile IR** 13.2 |
| Ada | sm_89 | L40, RTX 4090 | **Tile IR** 13.2 |
| Blackwell | sm_100 | B200 | **Tile IR** 13.1 |
| Blackwell | sm_120 | RTX 5090, RTX PRO 6000 | **Tile IR** 13.1 |

> **Note**
>
> Hopper (sm_90) is not supported in the 13.2 release.
