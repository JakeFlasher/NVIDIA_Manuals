---
title: "Current limitations"
section: ""
source: "https://docs.nvidia.com/cuda/cutile-python/execution.html#current-limitations"
---

#### [Current limitations](https://docs.nvidia.com/cuda/cutile-python#current-limitations)[](https://docs.nvidia.com/cuda/cutile-python/#current-limitations "Permalink to this headline")

The Python subset used in [tile code](https://docs.nvidia.com/cuda/cutile-python/#tile-code) imposes additional restrictions on control flow:

- `step` must be strictly positive.

Negative-step ranges such as
`range(10, 0, -1)` are not supported today. Passing a negative step
indirectly via a variable may lead to undefined behavior.
