---
title: "Constant Embedding"
section: ""
source: "https://docs.nvidia.com/cuda/cutile-python/execution.html#constant-embedding"
---

### [Constant Embedding](https://docs.nvidia.com/cuda/cutile-python#constant-embedding)[](https://docs.nvidia.com/cuda/cutile-python/#constant-embedding "Permalink to this headline")

If a parameter to a [kernel](https://docs.nvidia.com/cuda/cutile-python/#execution-tile-kernels) is _constant embedded_, then:

- All uses of the parameter shall act as if they were replaced by the literal value of the parameter.
- There shall be a distinct machine representation of the [kernel](https://docs.nvidia.com/cuda/cutile-python/#execution-tile-kernels) for each different value of the parameter that the [kernel](https://docs.nvidia.com/cuda/cutile-python/#execution-tile-kernels) is invoked with. Note: The [kernel](https://docs.nvidia.com/cuda/cutile-python/#execution-tile-kernels) shall be compiled once for each different value of the parameter, even if JIT caching is enabled.
- The [machine representation](https://docs.nvidia.com/cuda/cutile-python/interoperability.html#interoperability-machine-representation) of the parameter shall be 0 bytes.
