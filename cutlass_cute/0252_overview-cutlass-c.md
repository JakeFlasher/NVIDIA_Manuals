---
title: "CUTLASS C++"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/overview.html#overview--cutlass-c"
---

## [CUTLASS C++](https://docs.nvidia.com/cutlass/latest#cutlass-c)[](https://docs.nvidia.com/cutlass/latest/#cutlass-c "Permalink to this headline")

- Add [example 93](https://github.com/NVIDIA/cutlass/tree/main/examples/93_blackwell_low_latency_gqa/) for Blackwell low latency generation phase GQA kernel.
  - Flash Decoding with cluster reduction.
  - Kernel design details please check [Readme](https://github.com/NVIDIA/cutlass/tree/main/examples/93_blackwell_low_latency_gqa/readme.md).
- Add Blackwell SM100 State Space Decomposition (SSD) kernel in [example 112](https://github.com/NVIDIA/cutlass/tree/main/examples/112_blackwell_ssd).
- Add Hopper SM90 State Space Decomposition (SSD) kernel in [example 111](https://github.com/NVIDIA/cutlass/tree/main/examples/111_hopper_ssd).
- Add Hopper e2m1 to fp32 optimized conversion and e2m1 * TF32 tensor core GEMM.
  - Enable [example 55](https://github.com/NVIDIA/cutlass/tree/main/examples/55_hopper_mixed_dtype_gemm) with TF32 support
- Add [example 94](https://github.com/NVIDIA/cutlass/tree/main/examples/94_ada_fp8_blockwise/) for Ada FP8xFP8 -> BF16 GEMM with blockwise dequantization of input matrices in the MMA loop with FP32 accumulation.
- Add support for arbitrary application-provided strides for block-scale tensors.
  - Users and applications now must pass valid block-scale strides in all cases, even when the tensor is packed.
- Support 4x blockscaled public ptx for CUDA 13.1.
- Enable Blackwell SM120f compilation of examples and exposes NVFP4/MX Grouped GEMM in the CUTLASS Profiler.
- Allow non-static `TmaGbasis` in `AuxTmaParams`.
  - Some cases in attention kernel may require non-static `tma_gbasis`.
  - Relax the restriction on `TmaGbasis` parameter of `AuxTmaParams` and users are allowed to manually construct a dynamic gbasis.
- Fix some kernel issues:
  - Fix MSVC pre process issue.
  - Fix a self assign issue in GEMV kernel.
  - Fix a TMA descriptor bug where the CUDA driver is not properly setting the OOB address gen mode correctly.
  - Fix memory fence for clc scheduler in Blackwell SM120 pingpong kernel.
  - Fix missing SMEM alignment in Blackwell SM120 scale factors.
  - Fix a PDL issue for grouped gemm.
  - Fix divide-by-zero issue in canimplement for sm100 implicit gemm kernels.
  - Fix cluster swizzle for Grouped GEMMs.
    - Move host-side swizzling heuristics to device.
    - Apply swizzle per group based on problem shape and max swizzle size.
    - Improve examples and unit tests.
- Fix some profiler issues:
  - Fix a core dump issue for nvfp4 grouped GEMM kernel.
  - Fix inconsistent GEMM verification logic.
  - Rework grouped gemm verification logic for different types.
  - Fix api break change in using nvMatmulHeuristics.
- Fix some failed links under `media/docs`.

Note: CUTLASS 4.x builds are known to be down on Windows platforms for all CUDA toolkits.
CUTLASS team is working on a fix.

**See the [CHANGELOG](https://docs.nvidia.com/cutlass/latest/CHANGELOG.html) for details of all past releases and updates.**
