---
title: "Performance Profiling"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/overview.html#performance-profiling"
---

# [Performance Profiling](https://docs.nvidia.com/cutlass/latest#performance-profiling)[](https://docs.nvidia.com/cutlass/latest/#performance-profiling "Permalink to this headline")

The `tools/profiler/` directory contains a command-line utility for launching each of the GEMM kernels.
It can be built as follows:

```bash
$ make cutlass_profiler -j16
```
