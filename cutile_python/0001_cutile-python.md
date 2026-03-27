---
title: "cuTile Python"
section: ""
source: "https://docs.nvidia.com/cuda/cutile-python/index.html#cutile-python"
---

# [cuTile Python](https://docs.nvidia.com/cuda/cutile-python#cutile-python)[](https://docs.nvidia.com/cuda/cutile-python/#cutile-python "Permalink to this headline")

cuTile is a parallel programming model for NVIDIA GPUs and a Python-based DSL (Domain-Specific Language).
It automatically leverages advanced hardware capabilities, such as tensor cores and tensor memory accelerators,
while providing portability across different NVIDIA GPU architectures.
cuTile enables the latest hardware features without requiring code changes.

cuTile [kernels](https://docs.nvidia.com/cuda/cutile-python/execution.html#execution-tile-kernels) are GPU programs that are executed in parallel on a logical [grid](https://docs.nvidia.com/cuda/cutile-python/execution.html#grid) of [blocks](https://docs.nvidia.com/cuda/cutile-python/execution.html#block).
The [`@ct.kernel`](https://docs.nvidia.com/cuda/cutile-python/execution.html#cuda.tile.kernel "cuda.tile.kernel") decorator marks a Python function as a kernel’s entry point.
Kernels cannot be called directly from the host code; the host must queue kernels for execution on GPU
using the [`ct.launch()`](https://docs.nvidia.com/cuda/cutile-python/execution.html#cuda.tile.launch "cuda.tile.launch") function:

```python
import cuda.tile as ct
import cupy

TILE_SIZE = 16

# cuTile kernel for adding two dense vectors. It runs in parallel on the GPU.
@ct.kernel
def vector_add_kernel(a, b, result):
    block_id = ct.bid(0)
    a_tile = ct.load(a, index=(block_id,), shape=(TILE_SIZE,))
    b_tile = ct.load(b, index=(block_id,), shape=(TILE_SIZE,))
    result_tile = a_tile + b_tile
    ct.store(result, index=(block_id,), tile=result_tile)

# Host-side function that launches the above kernel.
def vector_add(a: cupy.ndarray, b: cupy.ndarray, result: cupy.ndarray):
    assert a.shape == b.shape == result.shape
    grid = (ct.cdiv(a.shape[0], TILE_SIZE), 1, 1)
    ct.launch(cupy.cuda.get_current_stream(), grid, vector_add_kernel, (a, b, result))
```

[Kernels](https://docs.nvidia.com/cuda/cutile-python/execution.html#execution-tile-kernels) move data between [arrays](https://docs.nvidia.com/cuda/cutile-python/data/array.html#data-array-cuda-tile-array) and [tiles](https://docs.nvidia.com/cuda/cutile-python/data.html#data-tiles-and-scalars) using functions like
[`ct.load()`](https://docs.nvidia.com/cuda/cutile-python/generated/cuda.tile.load.html#cuda.tile.load "cuda.tile.load") and [`ct.store()`](https://docs.nvidia.com/cuda/cutile-python/generated/cuda.tile.store.html#cuda.tile.store "cuda.tile.store").
Both arrays and tiles are tensor-like data structures: each has a specific shape
(i.e., the number of elements along each axis) and a [dtype](https://docs.nvidia.com/cuda/cutile-python/data.html#data-data-types) (i.e., the data type of elements).
However, there are important differences:

- [Arrays](https://docs.nvidia.com/cuda/cutile-python/data/array.html#data-array-cuda-tile-array) are stored in the global memory. They are mutable and have physical, strided
memory layouts. Within the kernel code, they support only a limited set of operations,
mostly related to [loading and storing](https://docs.nvidia.com/cuda/cutile-python/operations.html#operations-load-store) data to/from tiles. Various Python objects,
including PyTorch tensors and CuPy arrays, can be passed as arrays from the host code
to the kernel via kernel arguments.
- [Tiles](https://docs.nvidia.com/cuda/cutile-python/data.html#data-tiles-and-scalars) are immutable values without defined storage that only exist in the kernel code.
Tile dimensions must be compile-time constants that are powers of two.
Tiles support a multitude of [operations](https://docs.nvidia.com/cuda/cutile-python/operations.html#operations-operations), including elementwise arithmetic,
matrix multiplication, reduction, shape manipulation, etc.

Proceed to the [Quickstart](https://docs.nvidia.com/cuda/cutile-python/quickstart.html#quickstart) page for installation instructions and a complete working example.
