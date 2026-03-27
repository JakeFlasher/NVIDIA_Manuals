---
title: "2.1.3.3. Memory Management and Application Performance"
section: "2.1.3.3"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/intro-to-cuda-cpp.html#memory-management-and-application-performance"
---

### [2.1.3.3. Memory Management and Application Performance](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics#memory-management-and-application-performance)[](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/#memory-management-and-application-performance "Permalink to this headline")

As can be seen in the above example, explicit memory management is more verbose, requiring the programmer to specify copies between the host and device. This is the advantage and disadvantage of explicit memory management: it affords more control of when data is copied between host and devices, where memory is resident, and exactly what memory is allocated where. Explicit memory management can provide performance opportunities controlling memory transfers and overlapping them with other computations.

When using unified memory, there are CUDA APIs (which will be covered in [Memory Advise and Prefetch](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/understanding-memory.html#memory-mem-advise-prefetch)), which provide hints to the NVIDIA driver managing the memory, which can enable some of the performance benefits of using explicit memory management when using unified memory.
