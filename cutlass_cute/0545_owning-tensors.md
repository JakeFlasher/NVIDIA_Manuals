---
title: "Owning Tensors"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/03_tensor.html#owning-tensors"
---

### [Owning Tensors](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute#owning-tensors)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/#owning-tensors "Permalink to this headline")

A `Tensor` can also be an owning array of memory.
Owning `Tensor`s are created by calling `make_tensor<T>`,
where `T` is the type of each element of the array, and
a `Layout` or arguments to construct a `Layout`.
The array is allocated analogously to `std::array<T,N>` and, therefore, owning `Tensor`s must be constructed with a `Layout` that has static shapes and static strides.
CuTe does not perform dynamic memory allocation in `Tensor`s as it is not a common or performant operation within CUDA kernels.

Here are some examples of creating owning `Tensor`s.

```c++
// Register memory (static layouts only)
Tensor rmem_4x8_col = make_tensor<float>(Shape<_4,_8>{});
Tensor rmem_4x8_row = make_tensor<float>(Shape<_4,_8>{},
                                         LayoutRight{});
Tensor rmem_4x8_pad = make_tensor<float>(Shape <_4, _8>{},
                                         Stride<_32,_2>{});
Tensor rmem_4x8_like = make_tensor_like(rmem_4x8_pad);
```

The `make_tensor_like` function makes an owning Tensor of register memory with the same value type and shape as its input `Tensor` argument and attempts to use the same order of strides as well.

Calling `print` on each of the above tensors produces similar output

```console
rmem_4x8_col  : ptr[32b](0x7fff48929460) o (_4,_8):(_1,_4)
rmem_4x8_row  : ptr[32b](0x7fff489294e0) o (_4,_8):(_8,_1)
rmem_4x8_pad  : ptr[32b](0x7fff489295e0) o (_4,_8):(_32,_2)
rmem_4x8_like : ptr[32b](0x7fff48929560) o (_4,_8):(_8,_1)
```

and we can see that each pointer address is unique indicating that each `Tensor` is a unique array-like allocation.
