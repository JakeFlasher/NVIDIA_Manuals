---
title: "Object Model & Lifetimes"
section: ""
source: "https://docs.nvidia.com/cuda/cutile-python/execution.html#object-model-lifetimes"
---

### [Object Model & Lifetimes](https://docs.nvidia.com/cuda/cutile-python#object-model-lifetimes)[](https://docs.nvidia.com/cuda/cutile-python/#object-model-lifetimes "Permalink to this headline")

All objects created within [tile code](https://docs.nvidia.com/cuda/cutile-python/#tile-code) are immutable.
Any operation that conceptually modifies an object or its attributes creates and returns a new
object.
Attributes cannot be dynamically added to objects.

The only mutable objects that can be used in [tile code](https://docs.nvidia.com/cuda/cutile-python/#tile-code) are [arrays](https://docs.nvidia.com/cuda/cutile-python/data/array.html#data-array-cuda-tile-array), which must be passed in as
[kernel](https://docs.nvidia.com/cuda/cutile-python/#execution-tile-kernels) parameters.

The caller of a [kernel](https://docs.nvidia.com/cuda/cutile-python/#execution-tile-kernels) must ensure that:

- No [arrays](https://docs.nvidia.com/cuda/cutile-python/data/array.html#data-array-cuda-tile-array) passed to the [kernel](https://docs.nvidia.com/cuda/cutile-python/#execution-tile-kernels) alias one another.
- All passed [arrays](https://docs.nvidia.com/cuda/cutile-python/data/array.html#data-array-cuda-tile-array) remain valid until the [kernel](https://docs.nvidia.com/cuda/cutile-python/#execution-tile-kernels) has finished execution.
