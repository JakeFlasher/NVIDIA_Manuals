---
title: "CUTLASS 3.0 GEMM API"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/gemm_api_3x.html#cutlass-3-0-gemm-api"
---

# [CUTLASS 3.0 GEMM API](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#cutlass-3-0-gemm-api)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#cutlass-3-0-gemm-api "Permalink to this headline")

CUTLASS presents a uniform programming model
for matrix multiply-accumulate (MMA) operations
at different levels of the GPU system hierarchy.
CUTLASS 3.0 has GEMM APIs corresponding to the following levels
in order of highest to the lowest level.

1. Device
2. Kernel
3. Collective
4. Tiled MMA and Copy
5. Atom

This document will cover the first three levels in detail:
Device, Kernel, and Collective.
It also briefly discusses the Tiled MMA/Copy and Atom level,
and then refers readers to CuTe’s tutorial for more information.
