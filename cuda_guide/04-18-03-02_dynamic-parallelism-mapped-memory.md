---
title: "4.18.3.2. Mapped Memory"
section: "4.18.3.2"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/dynamic-parallelism.html#dynamic-parallelism--mapped-memory"
---

### [4.18.3.2. Mapped Memory](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#mapped-memory)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#mapped-memory "Permalink to this headline")

Mapped system memory has identical coherence and consistency guarantees to global memory, and follows the semantics detailed above. A kernel may not allocate or free mapped memory, but may use pointers to mapped memory passed in from the host program.
