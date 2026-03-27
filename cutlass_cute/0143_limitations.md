---
title: "Limitations"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/dsl_jit_caching.html#limitations"
---

### [Limitations](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general#limitations)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/#limitations "Permalink to this headline")

The intention of caching is to reduce the host launch overhead before each execution. As above example shows,
the consistency between the original Python code and the MLIR program is hard to maintain because of the impact of dynamic factors such as global variables.
Therefore, the MLIR program **MUST** always be generated to verify that the kernel content matches what was previously built.

For optimal host launch latency, we recommend using above custom caching method with `cute.compile`.
