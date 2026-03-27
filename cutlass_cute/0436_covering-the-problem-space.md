---
title: "Covering the problem space"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/profiler.html#covering-the-problem-space"
---

## [Covering the problem space](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#covering-the-problem-space)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#covering-the-problem-space "Permalink to this headline")

All arguments may have single values or comma-delimited set of values. Integers may also be specified
as an inclusive range with the following syntax `start:end:increment` or simply `start:end`.

For example, the following sweeps over the range of the GEMM K dimension from 8 to 4096 in increments
of 8 elements.

```bash
$ ./tools/profiler/cutlass_profiler --kernels=cutlass_simt_sgemm_128x128_nn --m=4352 --n=4096 --k=8:4096:8
```
