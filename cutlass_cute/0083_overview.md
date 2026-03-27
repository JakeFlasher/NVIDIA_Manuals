---
title: "Overview"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/overview.html#overview"
---

# [Overview](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL#overview)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/#overview "Permalink to this headline")

CUTLASS 4.x bridges the gap between productivity and performance for CUDA kernel development.
By providing Python-based DSLs to the powerful CUTLASS C++ template library, it enables
faster iteration, easier prototyping, and a gentler learning curve for high-performance linear
algebra on NVIDIA GPUs.

Overall we envision CUTLASS DSLs as a family of domain-specific languages (DSLs).
With the release of 4.0, we are releasing the first of these in CuTe DSL.
This is a low level programming model that is fully consistent with CuTe C++ abstractions — exposing
core concepts such as layouts, tensors, hardware atoms, and full control over the hardware thread and data hierarchy.
