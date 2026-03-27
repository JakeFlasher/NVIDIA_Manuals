---
title: "2.5.4.4. Link-Time Optimization (LTO)"
section: "2.5.4.4"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/nvcc.html#link-time-optimization-lto"
---

### [2.5.4.4. Link-Time Optimization (LTO)](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics#link-time-optimization-lto)[](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/#link-time-optimization-lto "Permalink to this headline")

[Separate compilation](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/#nvcc-separate-compilation) can result in lower performance than whole-program compilation due to limited cross-file optimization opportunities. Link-Time Optimization (LTO) addresses this by performing optimizations across separately compiled files at link time, at the cost of increased compilation time. LTO can recover much of the performance of whole-program compilation while maintaining the flexibility of separate compilation.

`nvcc` requires the `-dlto` [flag](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#dlink-time-opt-dlto) or `lto_<SM version>` link-time optimization targets to enable LTO:

```bash
nvcc -dc -dlto -arch=sm_100 definition.cu -o definition.o
nvcc -dc -dlto -arch=sm_100 example.cu    -o example.o
nvcc -dlto definition.o example.o -o program
```

```bash
nvcc -dc -arch=lto_100 definition.cu -o definition.o
nvcc -dc -arch=lto_100 example.cu    -o example.o
nvcc -dlto definition.o example.o -o program
```
