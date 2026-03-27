---
title: "4.1.3.3. Stream Associated Unified Memory"
section: "4.1.3.3"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/unified-memory.html#stream-associated-unified-memory"
---

### [4.1.3.3. Stream Associated Unified Memory](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#stream-associated-unified-memory)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#stream-associated-unified-memory "Permalink to this headline")

The CUDA programming model provides streams as a mechanism for programs to indicate dependence and independence among kernel launches. Kernels launched into the same stream are guaranteed to execute consecutively,
while kernels launched into different streams are permitted to execute concurrently.  See section [CUDA Streams](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html#cuda-streams).
