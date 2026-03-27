---
title: "3.2.4.1.2. Performance Considerations"
section: "3.2.4.1.2"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html#performance-considerations"
---

#### [3.2.4.1.2. Performance Considerations](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced#performance-considerations)[](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/#performance-considerations "Permalink to this headline")

- _Use the narrowest scope possible_: block-scoped atomics are much faster than system-scoped atomics.
- _Prefer weaker orderings_: use stronger orderings only when necessary for correctness.
- _Consider memory location_: shared memory atomics are faster than global memory atomics.
