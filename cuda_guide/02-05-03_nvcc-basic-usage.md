---
title: "2.5.3. NVCC Basic Usage"
section: "2.5.3"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/nvcc.html#nvcc-basic-usage"
---

## [2.5.3. NVCC Basic Usage](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics#nvcc-basic-usage)[](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/#nvcc-basic-usage "Permalink to this headline")

The basic command to compile a CUDA source file with `nvcc` is:

```bash
nvcc <source_file>.cu -o <output_file>
```

`nvcc` accepts common compiler flags used for specifying include directories `-I <path>` and library paths `-L <path>`, linking against other libraries `-l<library>`, and defining macros `-D<macro>=<value>`.

```bash
nvcc example.cu -I path_to_include/ -L path_to_library/ -lcublas -o <output_file>
```
