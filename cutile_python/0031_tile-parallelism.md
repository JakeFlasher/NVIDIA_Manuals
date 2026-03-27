---
title: "Tile Parallelism"
section: ""
source: "https://docs.nvidia.com/cuda/cutile-python/execution.html#tile-parallelism"
---

## [Tile Parallelism](https://docs.nvidia.com/cuda/cutile-python#tile-parallelism)[](https://docs.nvidia.com/cuda/cutile-python/#tile-parallelism "Permalink to this headline")

When a [block](https://docs.nvidia.com/cuda/cutile-python/#block) executes a function that takes [tiles](https://docs.nvidia.com/cuda/cutile-python/data.html#data-tiles-and-scalars) as parameters, it may parallelize the
evaluation of the function across the [block](https://docs.nvidia.com/cuda/cutile-python/#block)’s execution resources.
Unless otherwise specified, the execution shall complete before the function returns.
