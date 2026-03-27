---
title: "Global Arrays"
section: ""
source: "https://docs.nvidia.com/cuda/cutile-python/data.html#global-arrays"
---

## [Global Arrays](https://docs.nvidia.com/cuda/cutile-python#global-arrays)[](https://docs.nvidia.com/cuda/cutile-python/#global-arrays "Permalink to this headline")

A _global array_ (or _array_) is a container of elements of a specific [dtype](https://docs.nvidia.com/cuda/cutile-python/#data-data-types)
arranged in a logical multidimensional space.

Array’s _shape_ is a tuple of integer values, each denoting the length of
the corresponding dimension.
The length of the shape tuple equals the arrays’s number of dimensions.
The product of shape values equals the total logical number of elements in the array.

Arrays are stored in global memory using a _strided memory layout_: in addition to a shape,
an array also has an equally sized tuple of _strides_. Strides determine the mapping of logical
array indices to physical memory locations. For example, for a 3-dimensional *float32* array
with strides *(s1, s2, s3)*, the memory address of the element at the logical index
*(i1, i2, i3)* will be:

```default
base_addr + 4 * (s1 * i1 + s2 * i2 + s3 * i3),
```

where `base_addr` is the base address of the array and *4* is the byte size of a single *float32*
element.

New arrays can only be allocated by the host, and passed to the tile kernel as arguments.
[Tile code](https://docs.nvidia.com/cuda/cutile-python/execution.html#tile-code) can only create new views of existing arrays, for example using
[`Array.slice()`](https://docs.nvidia.com/cuda/cutile-python/data/array.html#cuda.tile.Array.slice "cuda.tile.Array.slice"). Like in Python, assigning an array object to another variable does not copy
the underlying data, but creates another reference to the array object.

Any object that implements the [DLPack](https://dmlc.github.io/dlpack/latest/) interface or the [CUDA Array Interface](https://numba.readthedocs.io/en/stable/cuda/cuda_array_interface.html)
can be passed to the kernel as an argument. Example: [CuPy](https://docs.cupy.dev/en/stable/) arrays and [PyTorch](https://pytorch.org/docs/stable/) tensors.

If two or more array arguments are passed to the kernel, their memory storage must not overlap.
Otherwise, behavior is undefined.

Array’s shape can be queried using the [`Array.shape`](https://docs.nvidia.com/cuda/cutile-python/data/array.html#cuda.tile.Array.shape "cuda.tile.Array.shape") attribute, which
returns a tuple of *int32* scalars. These scalars are non-constant, runtime values.
Using *int32* makes the tile code more performant at the cost of limiting the maximum
representable shape at 2,147,483,647 elements. This limitation will be lifted in the future.

> **See also**
>
> [cuda.tile.Array class documentation](https://docs.nvidia.com/cuda/cutile-python/data/array.html#data-array-cuda-tile-array)
