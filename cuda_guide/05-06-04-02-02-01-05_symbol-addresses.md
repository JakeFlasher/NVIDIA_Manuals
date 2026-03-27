---
title: "5.6.4.2.2.1.5. Symbol Addresses"
section: "5.6.4.2.2.1.5"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/device-callable-apis.html#symbol-addresses"
---

###### [5.6.4.2.2.1.5. Symbol Addresses](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#symbol-addresses)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#symbol-addresses "Permalink to this headline")

Device-side symbols (i.e., those marked `__device__`) may be referenced from within a kernel simply via the `&` operator, as all global-scope device variables are in the kernel’s visible address space. This also applies to `__constant__` symbols, although in this case the pointer will reference read-only data.

Since device-side symbols can be referenced directly, those CUDA runtime APIs which reference symbols (e.g., `cudaMemcpyToSymbol()` or `cudaGetSymbolAddress()`) are unnecessary and are not supported by the device runtime. This implies that constant data cannot be altered from within a running kernel, even ahead of a child kernel launch, as references to `__constant__` space are read-only.
