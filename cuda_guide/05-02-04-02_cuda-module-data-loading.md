---
title: "5.2.4.2. CUDA_MODULE_DATA_LOADING"
section: "5.2.4.2"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/environment-variables.html#cuda-module-data-loading"
---

### [5.2.4.2. CUDA_MODULE_DATA_LOADING](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#cuda-module-data-loading)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#cuda-module-data-loading "Permalink to this headline")

The environment variable affects how the CUDA runtime loads data associated to modules.

This is a complementary setting to the kernel-focused setting in `CUDA_MODULE_LOADING`. This environment variable does not affect the `LAZY` or `EAGER` loading of kernels.
Data loading behavior is inherited from `CUDA_MODULE_LOADING` if this environment variable is not set.

**Possible Values**:

- `DEFAULT`: Default behavior, equivalent to `LAZY`.
- `LAZY`: The loading of module data is delayed until a CUDA function handle, `CUfunc`, is required. In this case, the data from the CUBIN is loaded when the first kernel in the CUBIN is loaded or when the first variable in the CUBIN is accessed.
  - Lazy data loads can require context synchronization, which can slow down concurrent execution.
- `EAGER`: All data from a CUBIN, FATBIN, or PTX file are fully loaded upon the corresponding `cuModuleLoad*` and `cuLibraryLoad*` API call.

**Example**:

```bash
CUDA_MODULE_DATA_LOADING=EAGER
```
