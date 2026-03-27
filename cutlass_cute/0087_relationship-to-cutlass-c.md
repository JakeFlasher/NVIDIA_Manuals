---
title: "Relationship to CUTLASS C++"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/overview.html#relationship-to-cutlass-c"
---

# [Relationship to CUTLASS C++](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL#relationship-to-cutlass-c)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/#relationship-to-cutlass-c "Permalink to this headline")

CUTLASS DSLs are not a replacement for the CUTLASS C++ library or its 2.x and 3.x APIs. Instead, it aims to be a high-productivity kernel
authoring framework that shares all concepts with CUTLASS 3.x C++ API such as CuTe, pipelines, schedulers etc.

- **Performance**: Generated kernels aim to match CUTLASS C++ kernels in performance; however, some performance gaps
may exist due to missing optimizations that have been added over the years to CUTLASS C++ and may be missing in the DSLs examples.
- **Library**: The CUTLASS DSLs do not currently ship with a full GEMM/Conv autotuning profiler or library interface
akin to CUTLASS C++. Instead, it focuses on generating and autotuning individual kernel instances (for example: via tile size exploration) and via native integration DL frameworks that support auto-tuning.
