---
title: "4.1. Unified Memory"
section: "4.1"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/unified-memory.html#unified-memory--unified-memory"
---

# [4.1. Unified Memory](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#unified-memory)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#unified-memory "Permalink to this headline")

This section explains the detailed behavior and use of each of the different paradigms of unified memory available. [The earlier section on unified memory](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/understanding-memory.html#memory-unified-memory) showed how to determine which unified memory paradigm applies and briefly introduced each.

As discussed previously there are four paradigms of unified memory programming:

- [Full support for explicit managed memory allocations](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#um-pageable-systems)
- [Full support for all allocations with software coherence](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#um-pageable-systems)
- [Full support for all allocations with hardware coherence](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#um-pageable-systems)
- [Limited unified memory support](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#um-legacy-devices)

The first three paradigms involving full unified memory support have very similar behavior and programming model and are covered in [Unified Memory on Devices with Full CUDA Unified Memory Support](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#um-pageable-systems) with any differences highlighted.

The last paradigm, where unified memory support is limited, is discussed in detail in [Unified Memory on Windows, WSL, and Tegra](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#um-legacy-devices).
