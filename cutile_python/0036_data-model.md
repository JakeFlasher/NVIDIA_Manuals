---
title: "Data Model"
section: ""
source: "https://docs.nvidia.com/cuda/cutile-python/data.html#data-model"
---

# [Data Model](https://docs.nvidia.com/cuda/cutile-python#data-model)[](https://docs.nvidia.com/cuda/cutile-python/#data-model "Permalink to this headline")

cuTile is an array-based programming model.
The fundamental data structure is multidimensional arrays with elements of a single homogeneous type.
cuTile Python does not expose pointers, only arrays.

An array-based model was chosen because:

- Arrays know their bounds, so accesses can be checked to ensure safety and correctness.
- Array-based load/store operations can be efficiently lowered to speed-of-light hardware mechanisms.
- Python programmers are used to array-based programming frameworks such as NumPy.
- Pointers are not a natural choice for Python.

Within [tile code](https://docs.nvidia.com/cuda/cutile-python/execution.html#tile-code), only the types described in this section are supported.
