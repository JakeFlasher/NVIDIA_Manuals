---
title: "Tile Functions"
section: ""
source: "https://docs.nvidia.com/cuda/cutile-python/execution.html#tile-functions"
---

## [Tile Functions](https://docs.nvidia.com/cuda/cutile-python#tile-functions)[](https://docs.nvidia.com/cuda/cutile-python/#tile-functions "Permalink to this headline")

```
_`class`_`cuda.tile.``function`(_`func``=``None`_, _`/`_, _`*`_, _`host``=``False`_, _`tile``=``True`_)[](https://docs.nvidia.com/cuda/cutile-python/#cuda.tile.function "Link to this definition")
```

_Tile functions_ are functions that are usable in [tile code](https://docs.nvidia.com/cuda/cutile-python/#tile-code).

This decorator indicates what [execution spaces](https://docs.nvidia.com/cuda/cutile-python/#execution-execution-spaces) a function can be called from.
With no arguments, it denotes a tile-only function.

When an unannotated function is called by a [tile function](https://docs.nvidia.com/cuda/cutile-python/#execution-tile-functions), tile shall be added to the
unannotated function’s execution space.
This process is recursive.
No explicit annotation is required.

The types usable as parameters to a [tile function](https://docs.nvidia.com/cuda/cutile-python/#execution-tile-functions) are described in the [data model](https://docs.nvidia.com/cuda/cutile-python/data.html#data-data-model).

**Parameters:**
: - **host** (_bool__,__optional_) – Whether the function can be called from [host code](https://docs.nvidia.com/cuda/cutile-python/#host-code).
Default is False.
- **tile** (_bool__,__optional_) – Whether the function can be called from [tile code](https://docs.nvidia.com/cuda/cutile-python/#tile-code).
Default is True.
