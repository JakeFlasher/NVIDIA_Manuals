---
title: "2.4.2.3. Limited Unified Memory Support"
section: "2.4.2.3"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/understanding-memory.html#limited-unified-memory-support"
---

### [2.4.2.3. Limited Unified Memory Support](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics#limited-unified-memory-support)[](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/#limited-unified-memory-support "Permalink to this headline")

On Windows, including Windows Subsystem for Linux (WSL), and on some Tegra systems, a limited subset of unified memory functionality is available. On these systems, managed memory is available, but migration between CPU and GPUs behaves differently.

- Managed memory is first allocated in the CPU’s physical memory
- Managed memory is migrated in larger granularity than virtual memory pages
- Managed memory is migrated to the GPU when the GPU begins executing
- The CPU must not access managed memory while the GPU is active
- Managed memory is migrated back to the CPU when the GPU is synchronized
- Oversubscription of GPU memory is not allowed
- Only memory explicitly allocated by CUDA as managed memory is unified

Full coverage of this paradigm can be found in [Unified Memory on Windows, WSL, and Tegra](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/unified-memory.html#um-legacy-devices).
