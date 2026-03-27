---
title: "5.2.4.1. CUDA_MODULE_LOADING"
section: "5.2.4.1"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/environment-variables.html#cuda-module-loading"
---

### [5.2.4.1. CUDA_MODULE_LOADING](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#cuda-module-loading)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#cuda-module-loading "Permalink to this headline")

The environment variable affects how the CUDA runtime loads modules, specifically how it initializes the device code.

**Possible Values**:

- `DEFAULT`: Default behavior, equivalent to `LAZY`.
- `LAZY`: The loading of specific kernels is delayed until a CUDA function handle, `CUfunc`,  is extracted using the `cuModuleGetFunction()` or `cuKernelGetFunction()` API calls. In this case, the data from the CUBIN is loaded when the first kernel in the CUBIN is loaded or when the first variable in the CUBIN is accessed.
  - The driver loads the required code on the first call to a kernel; subsequent calls incur no extra overhead. This reduces startup time and GPU memory footprint.
- `EAGER`: Fully loads CUDA modules and kernels at program initialization. All kernels and data from a CUBIN, FATBIN, or PTX file are fully loaded upon the corresponding `cuModuleLoad*` and `cuLibraryLoad*` driver API call.
  - Higher startup time and GPU memory footprint. Kernel launch overhead is predictable.

**Examples**:

```bash
CUDA_MODULE_LOADING=EAGER
CUDA_MODULE_LOADING=LAZY
```
