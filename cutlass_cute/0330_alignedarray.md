---
title: "AlignedArray"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/fundamental_types.html#alignedarray"
---

### [AlignedArray](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#alignedarray)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#alignedarray "Permalink to this headline")

```c++
template <
  typename T,          // element type
  int N,               // number of elements
  int Alignment        // alignment requirement in bytes
>
class AlignedArray;
```

`AlignedArray` is derived from `Array<T, N>` and supports an optional alignment field. Pointers to objects of type `AlignedArray<>` reliably yield vectorized memory accesses when dereferenced.

Example:

```c++
int const kN = 8;
ArrayAligned<half_t, kN> source;
ArrayAligned<half_t, kN> const *ptr = ...;

source = *ptr;          // 128b aligned memory access
```
