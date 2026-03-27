---
title: "2.5.4.7. Compiler Performance Controls"
section: "2.5.4.7"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/nvcc.html#compiler-performance-controls"
---

### [2.5.4.7. Compiler Performance Controls](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics#compiler-performance-controls)[](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/#compiler-performance-controls "Permalink to this headline")

`nvcc` provides options to analyze and accelerate the compilation process itself:

- `-t <N>`: The number of CPU threads used to parallelize the compilation of a single compilation unit for multiple GPU architectures.
- `-split-compile <N>`: The number of CPU threads used to parallelize the optimization phase.
- `-split-compile-extended <N>`: More aggressive form of split compilation. Requires link-time optimization.
- `-Ofc <N>`: Level of device code compilation speed.
- `-time <filename>`: Generate a comma-separated value (CSV) table with the time taken by each compilation phase.
- `-fdevice-time-trace`: Generate a time trace for device code compilation.
