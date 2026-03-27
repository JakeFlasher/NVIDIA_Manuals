---
title: "2.5.4.2. Debugging Options"
section: "2.5.4.2"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/nvcc.html#debugging-options"
---

### [2.5.4.2. Debugging Options](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics#debugging-options)[](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/#debugging-options "Permalink to this headline")

`nvcc` supports the following options to generate debug information:

- `-g`: Generate debug information for host code. `gdb/lldb` and similar tools rely on such information for host code debugging.
- `-G`: Generate debug information for device code. [cuda-gdb](https://docs.nvidia.com/cuda/cuda-gdb/index.html) relies on such information for device-code debugging. The flag also defines the `__CUDACC_DEBUG__` macro.
- `-lineinfo`: Generate line-number information for device code. This option does not affect execution performance and is useful in conjunction with the [compute-sanitizer](https://developer.nvidia.com/compute-sanitizer) tool to trace the kernel execution.

`nvcc` uses the highest optimization level `-O3` for GPU code by default. The debug flag `-G` prevents some compiler optimizations, and so debug code is expected to have lower performance than non-debug code.  The `-DNDEBUG` flag can be defined to disable runtime assertions, as these can also slow down execution.
