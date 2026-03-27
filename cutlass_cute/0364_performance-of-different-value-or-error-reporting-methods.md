---
title: "Performance of different value-or-error reporting methods"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/programming_guidelines.html#performance-of-different-value-or-error-reporting-methods"
---

##### [Performance of different value-or-error reporting methods](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#performance-of-different-value-or-error-reporting-methods)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#performance-of-different-value-or-error-reporting-methods "Permalink to this headline")

[P1886R0](https://wg21.link/P1886R0)
(Ben Craig, “Error speed benchmarking”)
surveys different ways in Standard C++
to report errors from a function
that returns one or more values,
and compares their (host-only) performance
with different compilers.
