---
title: "5.4.6. Warp Functions"
section: "5.4.6"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#warp-functions"
---

## [5.4.6. Warp Functions](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#warp-functions)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#warp-functions "Permalink to this headline")

The following section describes the warp functions that allow threads within a warp to communicate with each other and perform computations.

> **Hint**
>
> It is suggested to use the `CUB` [Warp-Wide “Collective” Primitives](https://nvidia.github.io/cccl/cub/api_docs/warp_wide.html#warp-wide-collective-primitives)  to perform warp operations whenever possible for efficiency, safety, and portability reasons.
