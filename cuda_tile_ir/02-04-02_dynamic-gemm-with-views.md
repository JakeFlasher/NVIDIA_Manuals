---
title: "2.4.2. Dynamic GEMM with Views"
section: "2.4.2"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/prog_model.html#dynamic-gemm-with-views"
---

### [2.4.2. Dynamic GEMM with Views](https://docs.nvidia.com/cuda/tile-ir/latest/sections#dynamic-gemm-with-views)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#dynamic-gemm-with-views "Permalink to this headline")

We have now introduced the core concepts of **Tile IR**. We can put together many of these ideas
to support a dynamic GEMM kernel using structured pointers. To start, we make two changes to this GEMM:
the inputs are actually transposed into column-major layout, and the inputs are in `fp16` while the output is
in `fp32`.

We take the input and output tensors (as pointers), all dimension, and all strides as arguments.

```mlir
    entry @gemm_kloop_kernel(
        %A_ptr: !cuda_tile.tile<!cuda_tile.ptr<f16>>,
        %B_ptr: !cuda_tile.tile<!cuda_tile.ptr<f16>>,
        %C_ptr: !cuda_tile.tile<!cuda_tile.ptr<f32>>,
        %M: !cuda_tile.tile<i32>, %N: !cuda_tile.tile<i32>, %K: !cuda_tile.tile<i32>,
        %stride_ak: !cuda_tile.tile<i32>, %stride_bn: !cuda_tile.tile<i32>, %stride_cm: !cuda_tile.tile<i32>
```

The **Tile IR** compiler can optimize the memory loads and stores of [Tensor View](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tensor-view) if the
alignment of the underlying pointers and strides are known. If these are statically known then we can
infer the alignment directly but if they are dynamic, as they are in this case, we can use the
[cuda_tile.assume](https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html#op-cuda-tile-assume) operation to inform the compiler that the pointer is properly aligned.

Here we use the `div_by` predicate to inform the compiler about the divisibility of these
values which can be used to infer alignment constraints directly.

```mlir
%A_ptr_assume = assume #cuda_tile.div_by<16>, %A_ptr : tile<ptr<f16>>
%B_ptr_assume = assume #cuda_tile.div_by<16>, %B_ptr : tile<ptr<f16>>
%C_ptr_assume = assume #cuda_tile.div_by<16>, %C_ptr : tile<ptr<f32>>
%stride_ak_assume = assume #cuda_tile.div_by<8>, %stride_ak : tile<i32>
%stride_bn_assume = assume #cuda_tile.div_by<8>, %stride_bn : tile<i32>
%stride_cm_assume = assume #cuda_tile.div_by<8>, %stride_cm : tile<i32>
```

We create a tensor view for `%A`, `%B`, and `%C`.

```mlir
// A reference to the A tensor pointed to by A_ptr, (K x M)
%A = make_tensor_view %A_ptr_assume, shape = [%K, %M], strides = [%stride_ak, 1] : tile<i32> -> tensor_view<?x?xf16, strides=[?,1]>
// A reference to the B tensor pointed to by B_ptr, (N x K)
%B = make_tensor_view %B_ptr_assume, shape = [%N, %K], strides = [%stride_bn, 1] : tile<i32> -> tensor_view<?x?xf16, strides=[?,1]>
// A reference to the C tensor pointed to by C_ptr, (M x N)
%C = make_tensor_view %C_ptr_assume, shape = [%M, %N], strides = [%stride_cm, 1] : tile<i32> -> tensor_view<?x?xf32, strides=[?,1]>
```

We create a [Partition View](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-partition-view) for each tensor. First A:

```mlir
%A_block  = make_partition_view %A : partition_view<tile=(128x64), tensor_view<?x?xf16, strides=[?,1]>, dim_map=[1, 0]>
```

Then B:

```mlir
%B_block  = make_partition_view %B : partition_view<tile=(64x128), tensor_view<?x?xf16, strides=[?,1]>, dim_map=[1, 0]>
```

And last C:

```mlir
%C_block  = make_partition_view %C : partition_view<tile=(128x128), tensor_view<?x?xf32, strides=[?,1]>, dim_map=[0, 1]>
```

```mlir
    // Load a single 64x128 matrix from the tile.
    %B_frag, %t2 = load_view_tko weak %B_block [%k, %bidy] : partition_view<tile=(64x128), tensor_view<?x?xf16, strides=[?,1]>, dim_map=[1, 0]>, tile<i32> -> tile<64x128xf16>, token
```

We then read the the tile block grid coordinates using [cuda_tile.get_tile_block_id](https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html#op-cuda-tile-get-tile-block-id).

```mlir
%bidx, %bidy, %bidz = get_tile_block_id : tile<i32>
```

Due to the `K` dimension being dynamic we could do calculations ourselves to compute the size of the
index space of the reduction but **Tile IR** provides an operation to do this for us. [cuda_tile.get_index_space_shape](https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html#op-cuda-tile-get-index-space-shape)
to get the size of the index space, and thus the size of the reduction loop.

```mlir
%mk_len_i32:2 = get_index_space_shape %A_block : partition_view<tile=(128x64), tensor_view<?x?xf16, strides=[?,1]>, dim_map=[1, 0]> -> tile<i32>
```

No we can use a [cuda_tile.for](https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html#op-cuda-tile-for) loop to iterate over the tile blocks.

A `for` loop is one of the structured control flow operations in **Tile IR** it steps
a loop variable over a range from `(start, end, step)` executing the body for each value in the range.
The *iter_values* are loop carried variables of the body which are initialized in the loop header and are updated
each iteration by yielding them with the [cuda_tile.continue](https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html#op-cuda-tile-continue) operation.

In order to implement the reduction we start from `0` to the size of the tiled `K` dimension,
stepping uniformly by `1` on each step and a single loop carried variable representing the accumulator
tile.

```mlir
%result = for %k in (%i0 to %mk_len_i32#1, step %i1) : tile<i32>
    iter_values(%acc_prev = %cst) -> (tile<128x128xf32>)
```

We load tiles from A and B.

```mlir
// Load a single 128x64 matrix from the tile.
%A_frag, %t1 = load_view_tko weak %A_block[%bidx, %k] : partition_view<tile=(128x64), tensor_view<?x?xf16, strides=[?,1]>, dim_map=[1, 0]>, tile<i32> -> tile<128x64xf16>, token
// Load a single 64x128 matrix from the tile.
%B_frag, %t2 = load_view_tko weak %B_block [%k, %bidy] : partition_view<tile=(64x128), tensor_view<?x?xf16, strides=[?,1]>, dim_map=[1, 0]>, tile<i32> -> tile<64x128xf16>, token
```

We compute the MMA, and then continue to the next loop iteration with
the value.

```mlir
    %acc = mmaf %A_frag, %B_frag, %acc_prev: tile<128x64xf16>, tile<64x128xf16>, tile<128x128xf32>
    continue %acc : tile<128x128xf32>
```

Finally, like our previous implementation, outside the loop we store to the tile back to
`%C` this time via the view avoiding the need to compute the offsets again.

```mlir
%t3 = store_view_tko weak %result, %C_block[%bidx, %bidy] : tile<128x128xf32>, partition_view<tile=(128x128), tensor_view<?x?xf32, strides=[?,1]>, dim_map=[0, 1]>, tile<i32> -> token
```

Tensor views support dynamic shapes and strides, enabling robust, flexible kernels that handle varying input sizes
while maintaining high performance.
