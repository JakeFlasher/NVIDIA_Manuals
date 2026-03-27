---
title: "5.4.10.2. Breakpoint Function"
section: "5.4.10.2"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#breakpoint-function"
---

### [5.4.10.2. Breakpoint Function](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#breakpoint-function)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#breakpoint-function "Permalink to this headline")

The execution of a kernel function can be suspended by calling the `__brkpt()` function from any device thread.

```cuda
void __brkpt();
```
