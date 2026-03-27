---
title: "Memory Order"
section: ""
source: "https://docs.nvidia.com/cuda/cutile-python/memory_model.html#memory-order"
---

## [Memory Order](https://docs.nvidia.com/cuda/cutile-python#memory-order)[](https://docs.nvidia.com/cuda/cutile-python/#memory-order "Permalink to this headline")

```
_`class`_`cuda.tile.``MemoryOrder`[](https://docs.nvidia.com/cuda/cutile-python/#cuda.tile.MemoryOrder "Link to this definition")
```

Memory ordering semantics of an atomic operation.

```
`RELAXED`_`=` `'relaxed'`_[](https://docs.nvidia.com/cuda/cutile-python/#cuda.tile.MemoryOrder.RELAXED "Link to this definition")
```

No ordering guarantees. Cannot be used to synchronize between threads.

```
`ACQUIRE`_`=` `'acquire'`_[](https://docs.nvidia.com/cuda/cutile-python/#cuda.tile.MemoryOrder.ACQUIRE "Link to this definition")
```

Acquire semantics. When this reads a value written by a release,
the releasing thread’s prior writes become visible.
Subsequent reads/writes within the same block cannot be reordered before this operation.

```
`RELEASE`_`=` `'release'`_[](https://docs.nvidia.com/cuda/cutile-python/#cuda.tile.MemoryOrder.RELEASE "Link to this definition")
```

Release semantics. When an acquire reads the value written by this,
this thread’s prior writes become visible to the acquiring thread.
Prior reads/writes within the same block cannot be reordered after this operation.

```
`ACQ_REL`_`=` `'acq_rel'`_[](https://docs.nvidia.com/cuda/cutile-python/#cuda.tile.MemoryOrder.ACQ_REL "Link to this definition")
```

Combined acquire and release semantics.
