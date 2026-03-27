---
title: "Key Concepts and Approach"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/overview.html#key-concepts-and-approach"
---

# [Key Concepts and Approach](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL#key-concepts-and-approach)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/#key-concepts-and-approach "Permalink to this headline")

CUTLASS DSLs translate Python code into a custom intermediate representation (IR),
which is then Just-In-Time (JIT) compiled into optimized CUDA kernels using MLIR and *ptxas*.
