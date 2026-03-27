---
title: "2.4.1. Vector Addition with Views"
section: "2.4.1"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/prog_model.html#vector-addition-with-views"
---

### [2.4.1. Vector Addition with Views](https://docs.nvidia.com/cuda/tile-ir/latest/sections#vector-addition-with-views)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#vector-addition-with-views "Permalink to this headline")

We can now implement a more complex vector operation with SAXPY or “Single-Precision A·X Plus Y” (a common
BLAS operation). We can use `tensor view` and [cuda_tile.make_partition_view](https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html#op-cuda-tile-make-partition-view) to implement this operation for
arbitrary sized vectors.

The kernel first defines its arguments. We now take the inputs `%X`, `%Y`, `%alpha`,
as well as the dimensions of the **full** vectors, as arguments.

```mlir
entry @saxpy_memref(%X: tile<ptr<f32>>,
                    %Y: tile<ptr<f32>>,
                    %alpha: tile<f32>,
                    %M : tile<i32>,
                    %N : tile<i32>) {
```

For those familiar with other tile programming models, you might wonder why we need to take the input tensor sizes as arguments
instead of using them to control the number of blocks and remain implicit in the program.

The tensor view model differs from some existing tile programming models, wherein each kernel is unaware of the overall dimensions
of the problem size beyond the need to mask loads and stores. In contrast, `tensor view`’s require the overall tensor dimensions in order to enable
efficient and correct lowering of tiling and its related operations automatically.

```mlir
%x_memref = make_tensor_view %X, shape = [%M, %N], strides = [%M, 1] : tile<i32> -> tensor_view<?x?xf32, strides=[?,1]>
%y_memref = make_tensor_view %Y, shape = [%M, %N], strides = [%M, 1] : tile<i32> -> tensor_view<?x?xf32, strides=[?,1]>
```

The tensor view’s constructor takes the shapes and strides dynamically resulting in a dynamically shaped tensor view specified
by *?* in the type like so: `!cuda_tile.tile view<?x?xf32, strides=[?,1]>`.

We use [cuda_tile.make_partition_view](https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html#op-cuda-tile-make-partition-view) to create views for `%x` and `%y`
which represent a `(M/128 x N/256)` tensor of tiles. Each tile will have the size
specified by tile parameter of the partition type.

```mlir
%x_view = make_partition_view %x_memref : partition_view<tile=(128x256), tensor_view<?x?xf32, strides=[?,1]>>
%y_view = make_partition_view %y_memref : partition_view<tile=(128x256), tensor_view<?x?xf32, strides=[?,1]>>
```

[cuda_tile.load_view_tko](https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html#op-cuda-tile-load-view-tko) allows us to load the tile specified by `%view[%x, %y]`
for both *%X* and *%Y*.

```mlir
%x_tile, %token_x = load_view_tko weak %x_view[%tileIdX, %tileIdY] :
    partition_view<tile=(128x256), tensor_view<?x?xf32, strides=[?,1]>>, tile<i32> -> tile<128x256xf32>, token
```

We can then simply compute using the tiles directly to obtain our result tile.

```mlir
// Step 6. Compute sAXPY: y = alpha * A + y
```

Finally we accumulate the result into `%Y` directly. Here we treat it as both an input and
output parameter in this kernel.

```mlir

```

Combining tensor views with partitioning simplifies kernel implementation, allowing direct loading and operating on logical tiles
without manual offset arithmetic.
