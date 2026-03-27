---
title: "5.4.3.3. Maximum Number of Registers per Thread"
section: "5.4.3.3"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#maximum-number-of-registers-per-thread"
---

### [5.4.3.3. Maximum Number of Registers per Thread](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#maximum-number-of-registers-per-thread)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#maximum-number-of-registers-per-thread "Permalink to this headline")

To enable low-level performance tuning, CUDA C++ offers the `__maxnreg__()` function qualifier, which passes performance tuning information to the backend optimizing compiler. The `__maxnreg__()` qualifier specifies the maximum number of registers that can be allocated to a single thread in a thread block. In the definition of a `__global__` function:

```cuda
__global__ void
__maxnreg__(maxNumberRegistersPerThread)
MyKernel(...) {
    ...
}
```

The `maxNumberRegistersPerThread` variable specifies the maximum number of registers to be allocated to a single thread in a thread block of the kernel `MyKernel()`; it compiles to the `.maxnreg` PTX directive.

The `__launch_bounds__()` and `__maxnreg__()` qualifiers cannot be applied to the same kernel together.

The `--maxrregcount <N>` compiler option can be used to control register usage for all `__global__` functions in a file. This option is ignored for kernel functions with the `__maxnreg__` qualifier.
