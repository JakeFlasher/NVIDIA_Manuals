---
title: "A new Conceptual GEMM Hierarchy"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cutlass_3x_design.html#a-new-conceptual-gemm-hierarchy"
---

## [A new Conceptual GEMM Hierarchy](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#a-new-conceptual-gemm-hierarchy)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#a-new-conceptual-gemm-hierarchy "Permalink to this headline")

CUTLASS 2.x decomposes the moving parts of a GEMM operation
across a hierarchy that closely mirrors the organization of GPU
architectures. This discussed in detail within the
[CUTLASS 2.x GEMM API documentation](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/gemm_api.html).
This design, however, sometimes results in a coupling that is too tight
to extend to newer GPU features that might not fit into the same architectural
hierarchy. For instance, Hopper’s warp-group wide instructions do not naturally
fit into any warp or thread layer GEMM concept in CUTLASS 2.x. Even for Volta tensor cores,
instructions that atomically exist at the quad-pair granularity are first tiled at
the warp level before use. This hints at the brittleness of the abstraction power.

CUTLASS 3.0 detaches its interface layers from the hardware,
centering them instead around the natural structure of GEMM algorithms
not tied to any particular GPU generation.
This makes CUTLASS’s code more robust to GPU architecture evolution,
less prone to implementation detail leakage, and provides users
with a consistent interface to hardware acceleration regardless of
the architecture specific details.

The new conceptual GEMM hierarchy is discussed in detail in the dedicated
[CUTLASS 3.0 GEMM API documentation readme](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/gemm_api_3x.html),
along with code examples of the core concepts and types.
