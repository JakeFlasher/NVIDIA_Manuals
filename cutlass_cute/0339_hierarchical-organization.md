---
title: "Hierarchical Organization"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/programming_guidelines.html#hierarchical-organization"
---

## [Hierarchical Organization](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#hierarchical-organization)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#hierarchical-organization "Permalink to this headline")

The [CUTLASS 3.0 GEMM API](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/gemm_api_3x.html) document
explains CUTLASS 3.0’s hierarchical organization,
based conceptually on parallelization strategy.
This differs from CUTLASS 2.x’s approach,
which more closely mirrors the GPU hardware hierarchy
of thread blocks, warps, and threads.
