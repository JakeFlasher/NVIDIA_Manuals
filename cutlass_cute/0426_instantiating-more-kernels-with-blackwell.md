---
title: "Instantiating more kernels with Blackwell"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/profiler.html#instantiating-more-kernels-with-blackwell"
---

## [Instantiating more kernels with Blackwell](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#instantiating-more-kernels-with-blackwell)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#instantiating-more-kernels-with-blackwell "Permalink to this headline")

Blackwell (SM100) and Blackwell Ultra similarly support
`CUTLASS_LIBRARY_INSTANTIATION_LEVEL`, in order to instantiate all possible combinations.
Due to this, `CUTLASS_LIBRARY_KERNELS` must be non-empty, since generating and filtering these
kernels alone can take hours.
You must also exercise caution, because not all of these configs are tested, and some may fail to
compile or fail to launch at runtime.

```bash
$ cmake .. \
  -DCUTLASS_NVCC_ARCHS="100f" \
  -DCUTLASS_LIBRARY_KERNELS="cutlass3x_sm100_tensorop_gemm_f16_f16_f32_void_f32_*" \
  -DCUTLASS_LIBRARY_INSTANTIATION_LEVEL="max" \
  -DCUTLASS_UNITY_BUILD_ENABLED=ON
```

The CUTLASS profiler uses the same four-digit integer level (global instantiation level) mechanism to manage the generation of kernel configurations for Blackwell as well:

0. **Instruction Shape**
1. **MMA Shape Multiplier**
2. **Cluster Shape**
3. **Data Type and Schedule Pruning**

Note for Blackwell kernels an MMA shape multiplier is no longer necessary since Blackwell kernels do not have a different
ping pong or cooperative schedule. The profiler ignores this digit when instantiating.

Cluster shape levels define the number of CTAs (Cooperative Thread Arrays) included in the kernel generation:

- **Level 0**: Only dynamic cluster shapes.
- **Level 1**: For 1SM kernels `(1, 1, 1)` and `(2, 1, 1)` for 2SM kernels.
- **Level 2**: For 1SM kernels we also have `(1, 2, 1)` and for 2SM we have `(2, 2, 1)` and `(4, 1, 1)`.
- **Level 3**: For 1SM kernels we have `(1, 4, 1)` and for 2SM we have `(2, 4, 1)` and `(4, 2, 1)`.
- **Level 4**: For 1SM kernels we have `(4, 4, 1)` and for 2SM we have `(4, 4, 1)`.
- **Level 5**: For 1SM kernels we have `(2, 1, 1)`.
- **Level 6**: For 1SM kernels we have `(2, 2, 1)` and `(4, 1, 1)` and for 2SM kernels we have `(8, 1, 1)`.
- **Level 7**: For 1SM kernels we have `(2, 4, 1)` and `(4, 2, 1)`
- **Level 8**: For 1SM kernels we have `(1, 8, 1)` and `(8, 1, 1)`

Instruction shape levels control the selection of MMA shapes used in kernel generation:

- **Level 0**: Generates the “default” shape only.
- **Level 1**: Includes additional shapes for FP8, FP6, and FP4 as well as MX and NVFP4.
- **Level 2**: Includes small tile shapes.
- **Level 3**: Includes some non-power of 2 shapes.
- **Level 4**: Includes further small tile shapes and non-power of 2 shapes.
- **Level 5**: Includes all shapes.

The detailed definition of the three instantiation levels controlling cluster shape and instruction shape can be found in [sm100_shapes.py](https://github.com/NVIDIA/cutlass/tree/main/python/cutlass_library/sm100_shapes.py).
