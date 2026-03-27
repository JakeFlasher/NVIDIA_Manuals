---
title: "Design Limitations Likely to Remain"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/limitations.html#design-limitations-likely-to-remain"
---

## [Design Limitations Likely to Remain](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL#design-limitations-likely-to-remain)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/#design-limitations-likely-to-remain "Permalink to this headline")

The primary objective of CuTe DSL is to provide a domain-specific language for expressing
complex CUDA kernels with optimal GPU performance, not to execute arbitrary Python code on GPU hardware.

The following limitations will likely remain by design:

- **Complex Data Structures as Dynamic Values**: Lists, tuples, and dictionaries will continue to function
as static containers. While they can store dynamic values, their structure (adding/removing elements)
cannot be modified during execution of JIT-compiled functions.
- **Dependent Types**: Supporting dependent types would introduce substantial complexity and
adversely affect the performance characteristics of generated code.
- **CuTe Layout Algebra**: We don’t have plan to extend the support of CuTe Layout Algebra
under native Python Context. We are planning to extend support for data types and allow
JIT function to interoperate with native Python code.
