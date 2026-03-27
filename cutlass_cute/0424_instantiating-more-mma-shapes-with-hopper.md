---
title: "Instantiating more MMA shapes with Hopper"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/profiler.html#instantiating-more-mma-shapes-with-hopper"
---

## [Instantiating more MMA shapes with Hopper](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#instantiating-more-mma-shapes-with-hopper)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#instantiating-more-mma-shapes-with-hopper "Permalink to this headline")

When instantiating more tile shapes, specially non-power-of-2 Tile-N shapes, make sure to enable `CUTLASS_ENABLE_SM90_EXTENDED_MMA_SHAPES`.
This may lead to some increase in per-kernel compilation times.
When `CUTLASS_LIBRARY_INSTANTIATION_LEVEL` is set, then `CUTLASS_ENABLE_SM90_EXTENDED_MMA_SHAPES` is enabled by default.
