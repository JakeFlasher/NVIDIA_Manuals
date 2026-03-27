---
title: "5.2.2.4. CUDA_FORCE_PTX_JIT and CUDA_FORCE_JIT"
section: "5.2.2.4"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/environment-variables.html#cuda-force-ptx-jit-and-cuda-force-jit"
---

### [5.2.2.4. CUDA_FORCE_PTX_JIT and CUDA_FORCE_JIT](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#cuda-force-ptx-jit-and-cuda-force-jit)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#cuda-force-ptx-jit-and-cuda-force-jit "Permalink to this headline")

The environment variables instruct the CUDA driver to ignore any CUBIN embedded in an application and perform [Just-In-Time (JIT) compilation](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/cuda-platform.html#cuda-platform-just-in-time-compilation) of the embedded PTX code instead.

Forcing JIT compilation increases an application’s load time during initial execution. However, it can be used to validate that PTX code is embedded in an application and that its Just-In-Time compilation is functioning properly.
This ensures [forward compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/) with future architectures.

`CUDA_FORCE_PTX_JIT` overrides `CUDA_FORCE_JIT`.

**Possible Values**:

- `1`: Forces PTX JIT compilation.
- `0`: Default behavior.

**Example**:

```bash
CUDA_FORCE_PTX_JIT=1
```
