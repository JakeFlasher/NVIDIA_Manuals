---
title: "Mixed input data type kernels for Hopper"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/profiler.html#mixed-input-data-type-kernels-for-hopper"
---

## [Mixed input data type kernels for Hopper](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#mixed-input-data-type-kernels-for-hopper)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#mixed-input-data-type-kernels-for-hopper "Permalink to this headline")

With Hopper (SM90), the kernel generator will generate the following combinations of mixed input data types (“mixed dtype”):

| dtype(A) | dtype(B) |
| --- | --- |
| e4m3 | f16, bf16 |
| e5m2 | f16, bf16 |
| int8 | f16, bf16 |
| uint8 | f16, bf16 |
| int4 | f16, bf16 |
| int4 | e4m3, e5m2 |
| uint4 | f16, bf16 |
| int2 | f16, bf16 |
| uint2 | f16, bf16 |

For each mixed dtype kernel, the kernel generator will generate combinations of three different running modes:

- Convert-only
- Scale-only
- Scale-with-zero-point-shifting

For {4-bits-dtype, 8-bits-dtype} x 16-bits-dtype, the kernel generator will further generate kernels using shuffled layouts for the narrow data type matrix, which may have a better performance compared to its non-shuffle counter parts.
