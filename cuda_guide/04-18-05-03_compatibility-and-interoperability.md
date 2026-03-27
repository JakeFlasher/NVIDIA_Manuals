---
title: "4.18.5.3. Compatibility and Interoperability"
section: "4.18.5.3"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/dynamic-parallelism.html#compatibility-and-interoperability"
---

### [4.18.5.3. Compatibility and Interoperability](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#compatibility-and-interoperability)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#compatibility-and-interoperability "Permalink to this headline")

CDP2 is the default. Functions can be compiled with `-DCUDA_FORCE_CDP1_IF_SUPPORTED` to opt-out of using CDP2 on devices of compute capability less than 9.0.

|  | Function compiler with CUDA 12.0 and newer (default) | Function compiled with pre-CUDA 12.0 or with CUDA 12.0 and newer with `-DCUDA_FORCE_CDP1_IF_SUPPORTED` specified |
| --- | --- | --- |
| Compilation | Compile error if device code references `cudaDeviceSynchronize`. | Compile error if code references `cudaStreamTailLaunch` or `cudaStreamFireAndForget`. Compile error if device code references `cudaDeviceSynchronize` and code is compiled for sm_90 or newer. |
| Compute capability < 9.0 | New interface is used. | Legacy interface is used. |
| Compute capability 9.0 and higher | New interface is used. | New interface is used. If function references `cudaDeviceSynchronize` in device code, function load returns `cudaErrorSymbolNotFound` (this could happen if the code is compiled for devices of compute capability less than 9.0, but run on devices of compute capability 9.0 or higher using JIT). |

Functions using CDP1 and CDP2 may be loaded and run simultaneously in the same context. The CDP1 functions are able to use CDP1-specific features (e.g. `cudaDeviceSynchronize`) and CDP2 functions are able to use CDP2-specific features (e.g. tail launch and fire-and-forget launch).

A function using CDP1 cannot launch a function using CDP2, and vice versa. If a function that would use CDP1 contains in its call graph a function that would use CDP2, or vice versa, `cudaErrorCdpVersionMismatch` would result during function load.

The behavior of legacy CDP1 is not included in this document. For information on CDP1, refer back to a [legacy version of the CUDA Programming Guide](https://developer.nvidia.com/cuda-toolkit-archive)
