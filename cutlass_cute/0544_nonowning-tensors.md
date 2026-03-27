---
title: "Nonowning Tensors"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/03_tensor.html#nonowning-tensors"
---

### [Nonowning Tensors](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute#nonowning-tensors)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/#nonowning-tensors "Permalink to this headline")

A `Tensor` is usually a nonowning view of existing memory.
Nonowning `Tensor`s are created by calling `make_tensor`
with two arguments: a random-access iterator, and the `Layout` or arguments to construct a `Layout`.

Here are some examples of creating `Tensor`s
that are nonowning views of existing memory.

```cpp
float* A = ...;

// Untagged pointers
Tensor tensor_8   = make_tensor(A, make_layout(Int<8>{}));  // Construct with Layout
Tensor tensor_8s  = make_tensor(A, Int<8>{});               // Construct with Shape
Tensor tensor_8d2 = make_tensor(A, 8, 2);                   // Construct with Shape and Stride

// Global memory (static or dynamic layouts)
Tensor gmem_8s     = make_tensor(make_gmem_ptr(A), Int<8>{});
Tensor gmem_8d     = make_tensor(make_gmem_ptr(A), 8);
Tensor gmem_8sx16d = make_tensor(make_gmem_ptr(A), make_shape(Int<8>{},16));
Tensor gmem_8dx16s = make_tensor(make_gmem_ptr(A), make_shape (      8  ,Int<16>{}),
                                                   make_stride(Int<16>{},Int< 1>{}));

// Shared memory (static or dynamic layouts)
Layout smem_layout = make_layout(make_shape(Int<4>{},Int<8>{}));
__shared__ float smem[decltype(cosize(smem_layout))::value];   // (static-only allocation)
Tensor smem_4x8_col = make_tensor(make_smem_ptr(smem), smem_layout);
Tensor smem_4x8_row = make_tensor(make_smem_ptr(smem), shape(smem_layout), LayoutRight{});
```

As shown, users wrap the pointer by identifying its memory space:
e.g., global memory (via `make_gmem_ptr` or `make_gmem_ptr<T>`) or shared memory (via `make_smem_ptr` or `make_smem_ptr<T>`).
`Tensor`s that view existing memory can have either static or dynamic `Layout`s.

Calling `print` on all of the above tensors displays

```console
tensor_8     : ptr[32b](0x7f42efc00000) o _8:_1
tensor_8s    : ptr[32b](0x7f42efc00000) o _8:_1
tensor_8d2   : ptr[32b](0x7f42efc00000) o 8:2
gmem_8s      : gmem_ptr[32b](0x7f42efc00000) o _8:_1
gmem_8d      : gmem_ptr[32b](0x7f42efc00000) o 8:_1
gmem_8sx16d  : gmem_ptr[32b](0x7f42efc00000) o (_8,16):(_1,_8)
gmem_8dx16s  : gmem_ptr[32b](0x7f42efc00000) o (8,_16):(_16,_1)
smem_4x8_col : smem_ptr[32b](0x7f4316000000) o (_4,_8):(_1,_4)
smem_4x8_row : smem_ptr[32b](0x7f4316000000) o (_4,_8):(_8,_1)
```

which displays the pointer type along with any memory space tags, the pointer’s `value_type` width, the raw pointer address, and the associated `Layout`.
