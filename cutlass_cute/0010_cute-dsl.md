---
title: "CuTe DSL"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/CHANGELOG.html#cute-dsl"
---

### [CuTe DSL](https://docs.nvidia.com/cutlass/latest#cute-dsl)[](https://docs.nvidia.com/cutlass/latest/#cute-dsl "Permalink to this headline")

- New features
  - CuTe DSL now supports Python 3.14 for both x86_64 and aarch64
  - Runtime Pointer/Tensor/FakeTensor now supports **cache_key**, providing a stable, hashable representation that simplifies and improves compiled function caching.
- Bug fixing and improvements
  - Fixed Hopper FMHA causal attention performance regression on CUDA toolkit 13.1 by
optimizing mbarrier synchronization to avoid unnecessary convergence barriers.
  - Fix kernel loading race condition when multiple GPU are present in the same process in JAX.
