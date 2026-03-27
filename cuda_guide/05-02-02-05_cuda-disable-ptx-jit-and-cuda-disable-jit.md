---
title: "5.2.2.5. CUDA_DISABLE_PTX_JIT and CUDA_DISABLE_JIT"
section: "5.2.2.5"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/environment-variables.html#cuda-disable-ptx-jit-and-cuda-disable-jit"
---

### [5.2.2.5. CUDA_DISABLE_PTX_JIT and CUDA_DISABLE_JIT](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#cuda-disable-ptx-jit-and-cuda-disable-jit)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#cuda-disable-ptx-jit-and-cuda-disable-jit "Permalink to this headline")

The environment variables disable the [Just-In-Time (JIT) compilation](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/cuda-platform.html#cuda-platform-just-in-time-compilation) of embedded PTX code and use the compatible CUBIN embedded in an application.

A kernel will fail to load if it does not have embedded binary code, or if the embedded binary was compiled for an incompatible architecture. These environment variables can be used to validate that an application has compatible CUBIN code generated for each kernel. See the [Binary Compatibility](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/cuda-platform.html#cuda-platform-compute-binary-compatibility) section for more details.

`CUDA_DISABLE_PTX_JIT` overrides `CUDA_DISABLE_JIT`.

**Possible Values**:

- `1`: Disables PTX JIT compilation.
- `0`: Default behavior.

**Example**:

```bash
CUDA_DISABLE_PTX_JIT=1
```
