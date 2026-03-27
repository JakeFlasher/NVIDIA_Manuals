---
title: "Performance Tuning"
section: ""
source: "https://docs.nvidia.com/cuda/cutile-python/performance.html#performance-tuning"
---

# [Performance Tuning](https://docs.nvidia.com/cuda/cutile-python#performance-tuning)[](https://docs.nvidia.com/cuda/cutile-python/#performance-tuning "Permalink to this headline")

Several performance tuning techniques are available in cuTile:

- architecture-specific configuration values, using [`ByTarget`](https://docs.nvidia.com/cuda/cutile-python/#cuda.tile.ByTarget "cuda.tile.ByTarget");
- load/store hints such as `latency` and `allow_tma`.
