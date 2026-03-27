---
title: "2.2.3. GEMM Block by Block"
section: "2.2.3"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/prog_model.html#gemm-block-by-block"
---

### [2.2.3. GEMM Block by Block](https://docs.nvidia.com/cuda/tile-ir/latest/sections#gemm-block-by-block)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#gemm-block-by-block "Permalink to this headline")

Let us look at how to generalize this to work for a large matrix size – say `4096x4096`, where each tile block will
compute a single output tile of the final matrix. Note that we are still working with the assumption that our problem
shape is static and we will return to handling dynamically shaped inputs later on this section.

The kernel declaration looks identical to what we have been thus far taking scalar pointers to our inputs and outputs
as arguments.

```mlir
    entry @gemm_square_4096_tile_64x64_kernel(
        %a_ptr_base_scalar: tile<ptr<f32>>,
        %b_ptr_base_scalar: tile<ptr<f32>>,
        %c_ptr_base_scalar: tile<ptr<f32>>
```

In order to do full matrix multiplication over a large matrix we need to split the problem across multiple tile blocks
using the tile grid. We will use a 2-d grid and as such each thread will need to first read its 2-d block
coordinates.

```mlir
    // Read Tile block id's.
    %block_x_index, %block_y_index, %block_z_index = get_tile_block_id : tile<i32>
```

The block coordinates determine which tile of the output matrix we will be computing on this tile block. The kernel
is structured as a reduction over the K dimension of the matrix, producing individual but complete output tiles. It
loop over the reduction dimension computing a matrix-multiply-accumulate (MMA) operation over each input tile
that contributes to the output tile.

As we have been doing before the kernel begins by setting up the initial state. As this kernel is structured as a
loop we will first start by setting up the loop state.

```mlir
    %m_tile_size = constant <i32: 64> : tile<i32>
    %m_stride_factor = cuda_tile.constant <i32: 64> : tile<64x64xi32> // todo fix the line # and restore this %n_tile_size = cuda_tile.constant <i32: 64> : tile<i32>
    %k_tile_size = cuda_tile.constant <i32: 64> : tile<i32>
```

First we specify the tile sizes as constants. In this case because we are using square tiles of square matrices
everything is equivalent to 64. Constants in **Tile IR** can be tensor valued, and in this case we create 0-d
tensors containing a single scalar value.

```mlir
    %range_start = cuda_tile.constant <i32: 0> : tile<i32>
    %range_step = cuda_tile.constant <i32: 1> : tile<i32>
    %init_accum = cuda_tile.constant <f32: 0.000000e+00> : tile<64x64xf32>
```

Next we initialize constants for the loop which we will talk about more in a moment, and zero initialize an
accumulator value for the MMA. It is worth noting that the tensor here is a value, the **Tile IR** compiler
will deal with allocation and execution.

```mlir
    %tile_size_range = cuda_tile.iota : tile<64xi32>
```

We also create a shared range from `(0, 63)` using [cuda_tile.iota](https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html#op-cuda-tile-iota) again for the reduction dimension.

```mlir
    %a_tile_base = cuda_tile.muli %block_x_index, %k_tile_size : tile<i32>
    %a_tile_base_reshape = cuda_tile.reshape %a_tile_base : tile<i32> -> tile<1xi32>
    %a_tile_base_tensor = cuda_tile.broadcast %a_tile_base_reshape :
        tile<1xi32> -> tile<64xi32>
    %m_offsets_vec = cuda_tile.addi %a_tile_base_tensor, %tile_size_range : tile<64xi32>
```

We must first compute the tensor of offsets for A and B so that we can obtain a tensor of pointers for each in order to load
them from memory.

Conceptually we start by computing the offsets of the M dimension using the below Python pseudo code:

```python
m_offsets = block_x_index * k_tile_size + arange(0, k_tile_size)
```

This produces a vector starting from the “top-corner” of the tile at `(block_x_index * k_tile_size, block_x_index * k_tile_size + k_tile_size)`.

The striding of the A matrix is `(64, 1)`, meaning we can omit the extra multiplication by 1 for the K dimension
as each value in the row of the offset matrix is sequential.

For this the offsets in the K dimension would be `k_offs = arange(0, k_tile_size)`, so we can reuse
*%tile_size_range* below.

Again we can use the following Python pseudo code to compute the offset matrix:

```python
a_tile = reshape(m_offsets, (64, 1)) * m_stride + reshape(k_offsets, (1, 64)) * k_stride
```

In a libraries like Python’s NumPy this computation hides implicit broadcasting that must be expanded in **Tile IR**.

We will do this in two pieces by first computing the offsets along the `M` dimension, the relative offsets
along the `K` dimension, and then adding them together to produce the correct tensor of offsets.

We broadcast the `m_offsets` into a matrix where each column is identical and scaled by stride.

```mlir
%m_offsets_matrix = cuda_tile.reshape %m_offsets_vec :
    tile<64xi32> -> tile<64x1xi32>
%m_offsets_broadcast = cuda_tile.broadcast %m_offsets_matrix :
    tile<64x1xi32> -> tile<64x64xi32>
%m_offsets = cuda_tile.muli %m_offsets_broadcast, %m_stride_factor : tile<64x64xi32>
```

The matrix, which is now scaled by 64, would then contain these values:

```python
   [[   0,    0,    0,  ...,   0,    0,    0],
[  64,   64,   64,  ...,   64,   64,   64],
[ 128,  128,  128,  ...,  128,  128,  128],
   ...,
[3904, 3904, 3904,  ..., 3904,  3904,  3904],
[3968, 3968, 3968,  ..., 3968,  3968,  3968],
[4032, 4032, 4032,  ..., 4032,  4032,  4032]]
```

We then broadcast the k_offsets into a matrix where row is identical and scaled by stride.

```mlir
%ak_offsets_matrix = cuda_tile.reshape %tile_size_range :
     tile<64xi32> -> tile<1x64xi32>
%ak_offsets_broadcast = cuda_tile.broadcast %ak_offsets_matrix :
    tile<1x64xi32> -> tile<64x64xi32>
%ak_offsets = cuda_tile.muli %ak_offsets_broadcast, %m_stride_factor : tile<64x64xi32>
```

A is a row-major matrix (i.e the K dimension’s stride is 1) so the K offsets contains sequential values.

```python
   [[   0,    1,    2,  ...,   61,    62,    63],
[   0,    1,    2,  ...,   61,    62,    63],
[   0,    1,    2,  ...,   61,    62,    63],
   ...,
[   0,    1,    2,  ...,   61,    62,    63],
[   0,    1,    2,  ...,   61,    62,    63],
[   0,    1,    2,  ...,   61,    62,    63]]
```

We add the two of them together resulting in rows where the i-th row begins at the
i-th`value of :code:`m_offsets and each column contains the next 63 integers.

```mlir
%a_tile_offsets = cuda_tile.addi %m_offsets, %ak_offsets : tile<64x64xi32>
```

```python
   [[   0,    1,    2,  ...,   61,   62,   63],
[  64,   65,   66,  ...,  125,  126,  127],
[ 128,  129,  130,  ...,  189,  190,  191],
   ...,
[3904, 3905, 3906,  ..., 3965, 3966, 3967],
[3968, 3969, 3970,  ..., 4029, 4030, 4031],
[4032, 4033, 4034,  ..., 4093, 4094, 4095]]
```

Note that because this is the 0-th tile of the output matrix it is centered around (0, 0), the next tile will be
centered around (64, 0), then (128, 0) and so on. The above computation will result in the 64 x 64 grid at that
point based on the grid coordinates.

Now we must do the same for B.

```python
n_offs = j * n_tile_size + torch.arange(0, n_tile_size)
b_tile = k_offs[:, None] * k_stride + n_offs[None, :] * n_stride
```

```mlir
%b_tile_base = cuda_tile.muli %block_y_index, %k_tile_size : tile<i32>
%b_tile_base_reshape = cuda_tile.reshape %b_tile_base :
    tile<i32> -> tile<1xi32>
%b_tile_base_tensor = cuda_tile.broadcast %b_tile_base_reshape :
    tile<1xi32> -> tile<64xi32>
%n_offsets_vec = cuda_tile.addi %b_tile_base_tensor, %tile_size_range : tile<64xi32>
```

```mlir
%bk_offsets_matrix = cuda_tile.reshape %tile_size_range : tile<64xi32> -> tile<64x1xi32>
%bk_offsets = cuda_tile.broadcast %bk_offsets_matrix : tile<64x1xi32> -> tile<64x64xi32>
```

```mlir
%n_offsets_matrix = cuda_tile.reshape %n_offsets_vec : tile<64xi32> -> tile<1x64xi32>
%n_offsets_broadcast = cuda_tile.broadcast %n_offsets_matrix :  tile<1x64xi32> -> tile<64x64xi32>
%n_offsets = cuda_tile.muli %n_offsets_broadcast, %m_stride_factor : tile<64x64xi32>
%b_tile_offsets = cuda_tile.muli %bk_offsets, %n_offsets : tile<64x64xi32>
```

Now that we have computed the initial offsets for the pointers we simply convert the base pointers and
add the offsets like before.

```mlir
%a_ptr_base_tensor = cuda_tile.reshape %a_ptr_base_scalar :
    tile<ptr<f32>> -> tile<1x1xptr<f32>>
%a_ptr = cuda_tile.broadcast %a_ptr_base_tensor : tile<1x1xptr<f32>> -> tile<64x64xptr<f32>>
%a_tile_ptr = offset %a_ptr, %a_tile_offsets :
    tile<64x64xptr<f32>>, tile<64x64xi32> -> tile<64x64xptr<f32>>
```

Large matrix operations are decomposed into smaller tiles processed by a grid of blocks, requiring careful calculation of global
memory offsets for each block.
