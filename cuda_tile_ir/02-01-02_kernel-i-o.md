---
title: "2.1.2. Kernel I/O"
section: "2.1.2"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/prog_model.html#kernel-i-o"
---

### [2.1.2. Kernel I/O](https://docs.nvidia.com/cuda/tile-ir/latest/sections#kernel-i-o)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#kernel-i-o "Permalink to this headline")

To illustrate the design of **Tile IR** we will move from our simple hello world kernel
to one which implements 1-d tensor (i.e., vector) addition with a fixed block size `128`.
All examples presented in this section can be found in the [Programming Model Example Programs](https://docs.nvidia.com/cuda/tile-ir/latest/sections/appendix.html#section-appendix-sub-prog).

Tile kernels accept inputs and outputs as parameters; this is the only mechanism for consuming and producing data,
so we start by defining the kernel parameters.

```mlir
entry @vector_block_add_128x1_kernel(
    %a_ptr_base_scalar : !cuda_tile.tile<ptr<f32>>,
    %b_ptr_base_scalar : !cuda_tile.tile<ptr<f32>>,
    %c_ptr_base_scalar : !cuda_tile.tile<ptr<f32>>)
```

The above code fragment defines a kernel named `vector_block_add_128x1_kernel` which takes three arguments, representing
the two input buffers `a` and `b`, and the output buffer `c`. The types of all the arguments are scalar
pointers, which are represented as zero dimensional tensors containing a single pointer.

All values in **Tile IR** are either tensors, or tensor views (see [Tensor View](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-views)). A tensor is an n-dimensional rectangular array,
described by its rank (number of dimensions), the shape (extent along each dimension), and its primitive element type.
Tensors may have a rank of 0 or higher. Rank-0 tensors are scalars. The rank, dimensions, and element type are all part
of the tensor type and are statically known. Tensor types are assigned to values which represent a logical view of a
multidimensional array contained in global memory. Global memory is always accessed via tensors, and thus
pointer arguments always point to CUDA device allocations in global device memory. Tile kernels do not have return values
and thus omit a return type annotation (note that tile functions may have a return type, and will be discussed later).

An astute reader might now be wondering why we gave the inputs and outputs scalar pointer types instead of
tensors types with statically known rank, shape, and stride. A common pattern in the **Tile IR** programming model
is for kernels to take unstructured base pointers as parameters which can then be used to construct the required tensor.
This flexibility gives rise to multiple ways to use pointers depending on the desired program behavior, but we will first
focus on the most flexible representation by converting our base pointer into a tensor of arbitrary pointers.
A tensor of arbitrary pointers is a flexible abstraction which allows you to perform scatter/gather style loads
from a set of addresses all at once, and treat a set of addresses as a logical tile.

We must take a few steps to convert a base pointer
`%a_ptr_base_scalar` into a tensor of pointers representing the `128x1` tile we want to compute
on which contains addresses from `(base + 0, ..., base + 127)`

We start by creating an offset tensor which represents the inclusive `(0, 127)` interval.
We use [cuda_tile.iota](https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html#op-cuda-tile-iota) which constructs a range tensor that counts from `0` to `n - 1`
forming an `n` sized vector.

```mlir
    %offset = iota : tile<128xi32>
```

We then reshape from a scalar `ptr<f32>` to a 1-d tensor `1xptr<f32>` so we have the correct rank.

```mlir
%a_ptr_base_tensor = reshape %a_ptr_base_scalar :
    tile<ptr<f32>> -> tile<1xptr<f32>>
```

We then broadcast the pointer so we have a 1-d tensor of `(base, ..., base)` containing 128 elements.

```mlir
%a_ptr = broadcast %a_ptr_base_tensor : tile<1xptr<f32>> -> tile<128xptr<f32>>
```

Add the offset tensor to the tensor of pointers to obtain a `tile<128xptr<f32>>` that contains
pointers of `(base + 0, ..., base + 127)` as its values. Now we have a tensor of pointers which represents
the tile that we would like to compute on. We perform the same set of steps for `a`, `b`, and `c`.

```mlir
%a_tensor = offset %a_ptr, %offset :
    tile<128xptr<f32>>, tile<128xi32> -> tile<128xptr<f32>>
```

Finally, we load both operands, perform the addition, and store to the output.

```mlir
%a_val, %token_a = load_ptr_tko weak %a_tensor : tile<128xptr<f32>> -> tile<128xf32>, token
%b_val, %token_b = load_ptr_tko weak %b_tensor : tile<128xptr<f32>> -> tile<128xf32>, token
%c_val = addf %a_val, %b_val rounding<nearest_even> : tile<128xf32>
store_ptr_tko weak %c_tensor, %c_val : tile<128xptr<f32>>, tile<128xf32> -> token
```

We now have a complete
kernel for a single tile-block that performs addition over 128 element vectors. As you can see this code is
written from a single thread of control, but its level of parallelism will be determined by the compiler.

Kernels communicate with global memory via pointer arguments, which can be transformed into tensors using shape and stride
manipulation for flexible data access.
