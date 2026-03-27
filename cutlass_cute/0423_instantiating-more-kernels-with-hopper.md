---
title: "Instantiating more kernels with Hopper"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/profiler.html#instantiating-more-kernels-with-hopper"
---

## [Instantiating more kernels with Hopper](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#instantiating-more-kernels-with-hopper)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#instantiating-more-kernels-with-hopper "Permalink to this headline")

With Hopper (SM90), you will need to use an additional flag,
`CUTLASS_LIBRARY_INSTANTIATION_LEVEL`, in order to instantiate all possible combinations,
which unlike previous architectures, will be in the order of millions of kernels.
Due to this, `CUTLASS_LIBRARY_KERNELS` must be non-empty, since generating and filtering these
kernels alone can take hours.
You must also exercise caution, because not all of these configs are tested, and some may fail to
compile or fail to launch at runtime.

```bash
$ cmake .. \
  -DCUTLASS_NVCC_ARCHS="90a" \
  -DCUTLASS_LIBRARY_KERNELS="cutlass3x_sm90_tensorop_gemm_f16_f16_f32_void_f32_*" \
  -DCUTLASS_LIBRARY_INSTANTIATION_LEVEL="max" \
  -DCUTLASS_UNITY_BUILD_ENABLED=ON
```

The CUTLASS profiler employs a four-digit integer level (global instantiation level) mechanism to manage the generation of kernel configurations. This global instantiation level decides the behavior of multiple “generators” by defining how many and which combinations of configurations are produced. If a global instantiation level contains fewer than four digits, it can be padded with leading zeros to ensure it is four digits long. Each of the four digits in the global level corresponds to a specific category that influences kernel generation, from right to left:

0. **Instruction Shape**
1. **MMA Shape Multiplier**
2. **Cluster Shape**
3. **Schedule Pruning**

Cluster shape levels define the number of CTAs (Cooperative Thread Arrays) included in the kernel generation:

- **Level 0**: Only `(1, 2, 1)` cluster shape.
- **Level 1**: Clusters with 2 CTAs.
- **Level 2**: Clusters with 1 or 2 CTAs.
- **Level 3**: Clusters with 1, 2, or 4 CTAs.
- **Level 4**: Clusters with 1, 2, 4, or 8 CTAs.
- **Level 5**: Clusters with 1, 2, 4, 8, or 16 CTAs.

The MMA multipliers are combined with MMA instruction shapes (WGMMA shapes) to form CTA shapes. The levels for MMA multipliers determine the configurations generated for different data types.

- **Levels [0, 3]**: Control the specific configurations generated for various data types.
- **Level 9**: Activates exhaustive mode, generating all possible configurations.

Higher levels encompass a broader range of CTA configurations, resulting in more comprehensive kernel generation.

Instruction shape levels control the selection of WGMMA shapes used in kernel generation:

- **Level 0**: Generates the “default” shape only.
- **Level 1**: Includes additional shapes for unpruned cases, specifically for TF32 data type.
- **Level 2**: Includes shapes that are powers of 2.
- **Level 3**: Includes all other shapes.

The detailed definition of the three instantiation levels controlling cluster shape, MMA shape multiplier, and instruction shape can be found in [sm90_shapes.py](https://github.com/NVIDIA/cutlass/tree/main/python/cutlass_library/sm90_shapes.py).

Schedule pruning levels decide the epilogue schedule and mainloop schedule to stamp out a kernel instance. As defined in `get_valid_schedules` in [sm90_utils.py](https://github.com/NVIDIA/cutlass/tree/main/python/cutlass_library/sm90_utils.py),

- **Level >= 1**: Indicates that no pruning is being applied.
- **Level 0**: Indicates pruning according to existing [generator.py](https://github.com/NVIDIA/cutlass/tree/main/python/cutlass_library/generator.py) behavior.

An instantiation level `500`, which is padded to `0500`, thus indicates:

- **Instruction Shapes**: At level 0, generating only the “default” shape.
- **MMA Multipliers**: At level 0, generating only one multiplier, `(2, 1, 4)`.
- **Cluster Sizes**: At level 5, allowing for clusters with 1, 2, 4, 8, or 16 CTAs.
- **Schedule Pruning**: At level 0, where pruning is applied according to the existing `generator.py` behavior.
