---
title: "4.2.0 (2025-09-10)"
section: "4.2.0"
source: "https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/changelog.html#changelog--id4"
---

## [4.2.0 (2025-09-10)](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api#id4)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#id4 "Permalink to this headline")

- Added back `cute.make_tiled_copy` per the request from community
- Added support for explicit and implicit broadcast in `TensorSSA`
  - `cutlass.cute.TensorSSA`: support `broadcast_to` and implicit broadcasting for binary operations.
- Supported printing `TensorSSA` value in `cutlass.cute.print_tensor`
- Updated `cute.gemm` to support all dispatch patterns and improved checks for illegal inputs
- Introduced automatic kernel smem usage calculation for launch config.
- Introduced per op fast-math control for math ops(e.g. `exp`, `exp2`, `log2`, `log`)
- Introduced `CopyReduceBulkTensorTileS2GOp` in [tcgen05/copy.py](https://github.com/NVIDIA/cutlass/blob/main/python/CuTeDSL/cutlass/cute/nvgpu/tcgen05/copy.py) to support TMA Reduce.
