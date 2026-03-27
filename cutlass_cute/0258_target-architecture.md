---
title: "Target Architecture"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/overview.html#target-architecture"
---

## [Target Architecture](https://docs.nvidia.com/cutlass/latest#target-architecture)[](https://docs.nvidia.com/cutlass/latest/#target-architecture "Permalink to this headline")

In general, PTX code generated for one target architecture can be run on future architectures
(i.e., it is forward compatible).
However, CUDA 12.0 introduced the concept of “architecture-accelerated features” whose
PTX does not have forward compatibility guarantees.
Several Hopper and Blackwell PTX instructions fall under this category of
architecture-accelerated features, and thus require a `sm_90a` or `sm100a` target architecture
(note the “a” appended). For more details on this and other architecture-accelerated instructions,
please refer to the [CUDA Documentation](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#feature-availability).

The target architecture information is passed on to CUTLASS via the cmake flag
`CUTLASS_NVCC_ARCHS`. In order to maximize performance on Hopper GH100,
users are required to build CUTLASS with `90a` as the target architecture.
If a user accidentally builds a kernel which uses SM90a features
(e.g. Hopper Tensor Core Instructions), using the SM90 target
(note the lack of “a”), with either CUDA Toolkit 12 or 11.8,
the kernel is expected to fail with a runtime error.

```console
cmake .. -DCUTLASS_NVCC_ARCHS="90a"
```

Or

```console
cmake .. -DCUTLASS_NVCC_ARCHS="100a"
```

Note: The NVIDIA Blackwell SM100 architecture used in the datacenter
products has a different compute capability than the one underpinning
NVIDIA Blackwell GeForce RTX 50 series GPUs (SM120). As a result, kernels
compiled for Blackwell SM100 architecture with arch conditional features
(using `sm100a`) are not compatible with RTX 50 series GPUs.

Please refer to the [functionality documentation](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/functionality.html)
for details on which kernels require which target architectures.
