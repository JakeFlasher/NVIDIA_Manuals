---
title: "4.5. Programmatic Dependent Launch and Synchronization"
section: "4.5"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html#programmatic-dependent-launch-and-synchronization"
---

# [4.5. Programmatic Dependent Launch and Synchronization](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#programmatic-dependent-launch-and-synchronization)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#programmatic-dependent-launch-and-synchronization "Permalink to this headline")

The _Programmatic Dependent Launch_ mechanism allows for a dependent _secondary_ kernel
to launch before the _primary_ kernel it depends on in the same CUDA stream has finished executing.
Available starting with devices of compute capability 9.0, this technique can provide performance
benefits when the _secondary_ kernel can complete significant work that does not depend on the results of the _primary_ kernel.
