---
title: "Machine Representation"
section: ""
source: "https://docs.nvidia.com/cuda/cutile-python/interoperability.html#machine-representation"
---

## [Machine Representation](https://docs.nvidia.com/cuda/cutile-python#machine-representation)[](https://docs.nvidia.com/cuda/cutile-python/#machine-representation "Permalink to this headline")

cuTile executes Python [tile code](https://docs.nvidia.com/cuda/cutile-python/execution.html#tile-code) on NVIDIA GPUs by translating the Python code into a _machine representation_ that can be executed by CUDA devices.
Functions, types, and objects all have a machine representation.

Machine representations are defined in terms of corresponding CUDA C++ entities.
Example: `cuda.tile.float16` has the same machine representation as `__half` in CUDA C++.
