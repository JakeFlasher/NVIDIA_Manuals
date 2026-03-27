---
title: "Memory Scope"
section: ""
source: "https://docs.nvidia.com/cuda/cutile-python/memory_model.html#memory-scope"
---

## [Memory Scope](https://docs.nvidia.com/cuda/cutile-python#memory-scope)[](https://docs.nvidia.com/cuda/cutile-python/#memory-scope "Permalink to this headline")

```
_`class`_`cuda.tile.``MemoryScope`[](https://docs.nvidia.com/cuda/cutile-python/#cuda.tile.MemoryScope "Link to this definition")
```

The scope of threads that participate in memory ordering.

```
`BLOCK`_`=` `'block'`_[](https://docs.nvidia.com/cuda/cutile-python/#cuda.tile.MemoryScope.BLOCK "Link to this definition")
```

Ordering guarantees apply to threads within the same block.

```
`DEVICE`_`=` `'device'`_[](https://docs.nvidia.com/cuda/cutile-python/#cuda.tile.MemoryScope.DEVICE "Link to this definition")
```

Ordering guarantees apply to all threads on the same GPU.

```
`SYS`_`=` `'sys'`_[](https://docs.nvidia.com/cuda/cutile-python/#cuda.tile.MemoryScope.SYS "Link to this definition")
```

Ordering guarantees apply to all threads across the entire system,
including multiple GPUs and the host.
