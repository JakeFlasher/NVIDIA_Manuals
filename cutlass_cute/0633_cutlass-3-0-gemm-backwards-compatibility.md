---
title: "CUTLASS 3.0 GEMM Backwards Compatibility"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cutlass_3x_backwards_compatibility.html#cutlass-3-0-gemm-backwards-compatibility"
---

# [CUTLASS 3.0 GEMM Backwards Compatibility](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#cutlass-3-0-gemm-backwards-compatibility)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#cutlass-3-0-gemm-backwards-compatibility "Permalink to this headline")

Although CUTLASS 3.0 restructures the GEMM hierarchy and introduces new types for the
threadblock layer and below, we intend the entire source code to be usable in user applications.
We expect users to be able to `#include` any source file from CUTLASS 3.0, whether
they implement the 2.x or the 3.x API, without breaking user builds. This means that a single
translation unit should be able to contain any valid kernel regardless of its API version. The
sections below discuss how `device` and `kernel` layer type names are made compatible across the
two API versions, and what the users can expect out of the `threadblock` layer API going forward.
