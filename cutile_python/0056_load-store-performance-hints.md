---
title: "Load/store performance hints"
section: ""
source: "https://docs.nvidia.com/cuda/cutile-python/performance.html#load-store-performance-hints"
---

## [Load/store performance hints](https://docs.nvidia.com/cuda/cutile-python#load-store-performance-hints)[](https://docs.nvidia.com/cuda/cutile-python/#load-store-performance-hints "Permalink to this headline")

The [`load()`](https://docs.nvidia.com/cuda/cutile-python/generated/cuda.tile.load.html#cuda.tile.load "cuda.tile.load") and [`store()`](https://docs.nvidia.com/cuda/cutile-python/generated/cuda.tile.store.html#cuda.tile.store "cuda.tile.store") operations accept optional keyword
arguments that can influence how memory traffic is scheduled and lowered:

- `latency` (`int` or `None`) – A hint indicating how heavy the DRAM
traffic will be for this operation. It shall be an integer between
1 (low) and 10 (high). If it is `None` or not provided, the compiler
will infer the latency.
- `allow_tma` (`bool` or `None`) – If `True`, the load or store may be
lowered to use TMA (Tensor Memory Accelerator) when the target architecture
supports it. If `False`, TMA will not be used for this operation.
By default, TMA is allowed.

These hints are optional: kernels will compile and run without specifying
them, but providing them can help the compiler make better code-generation
decisions for a particular memory-access pattern.
