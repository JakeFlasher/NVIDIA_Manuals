---
title: "Execution Spaces"
section: ""
source: "https://docs.nvidia.com/cuda/cutile-python/execution.html#execution-spaces"
---

## [Execution Spaces](https://docs.nvidia.com/cuda/cutile-python#execution-spaces)[](https://docs.nvidia.com/cuda/cutile-python/#execution-spaces "Permalink to this headline")

cuTile code is executed on one or more _targets_, which are distinct execution environments that
are distinguished by different hardware resources or programming models.

A function is _usable_ if it can be called.
A type or object is _usable_ if its attributes are accessible (can be read and written) and its
methods are callable.

Some functions, types, and objects are only usable on certain _targets_.
The set of _targets_ that such a construct is usable on is called its _execution space_.

- _Host code_ is the execution space that includes all CPU targets.
- _SIMT code_ is the execution space that includes all CUDA SIMT targets.
Note: This has historically been called device code, but we avoid this term to prevent ambiguity.
- _Tile code_ is the execution space that includes all CUDA tile targets.

Functions can have decorators that explicitly specify their execution space.
These are called _annotated functions_.
