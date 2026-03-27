---
title: "AlignedBuffer"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/fundamental_types.html#alignedbuffer"
---

### [AlignedBuffer](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#alignedbuffer)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#alignedbuffer "Permalink to this headline")

```c++
template <
  typename T,          // element type
  int N,               // number of elements
  int Alignment        // alignment requirement in bytes
>
class AlignedBuffer;
```

`AlignedBuffer` provides a uniform way to define aligned memory allocations for all data types. This is particularly
useful in defining allocations within shared memory with guaranteed memory alignment needed for vectorized access.
Note, constructors of the elements within AlignedBuffer<> are not called, and so the elements are initially in an
undefined state.

Use `AlignedBuffer<>::data()` to obtain a pointer to the first element of the buffer.

**Example:** Guaranteed aligned shared memory allocation. Note, shared memory contents are uninitialized.

```c++
int const kN = 32;
int const kAlignment = 16;                  // alignment in bytes

// Define a shared memory allocation in device code
__shared__ AlignedBuffer<complex<half_t>, kN, kAlignment> matrix_tile;

complex<half_t> *ptr = matrix_tile.data();  // ptr is guaranteed to have 128b (16 Byte) alignment
```

Note, `AlignedBuffer<>` only guarantees that its internal memory allocation is aligned, obtained by `AlignedBuffer<>::data()`. There is no guarantee that the `AlignedBuffer<>` object itself satisfies alignment constraints or that its internal memory allocation is contiguous. Device code performing vectorized memory accesses should use the `AlignedArray<>` type.

**_Example_:** Vectorized memory access to shared memory allocations.

```c++
int const kN = 1024;

__shared__ AlignedBuffer<half_t, kN> smem_buffer;

AlignedArray<half_t, 8> *ptr = reinterpret_cast<AlignedArray<half_t, 8> *>(smem_buffer.data());

AlignedArray<half_t, 8> x = ptr[threadIdx.x];     // 128b shared memory load
```
