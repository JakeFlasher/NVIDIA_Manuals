---
title: "DivBy"
section: ""
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html#divby"
---

#### [DivBy](https://docs.nvidia.com/cuda/tile-ir/latest/sections#divby)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#divby "Permalink to this headline")

```mlir
div_by< $divisor (, every $every^ along $along)?>
```

The `div_by` attribute must be used as a predicate for `cuda_tile.assume`
ops. The predicated value must be a `tile` of integers or pointers, or
a `tensor_view`.

If the predicated value is a `tile`, the attribute indicates that some
elements of the `tile` are divisible by `divisor`. If the predicated value
is a `tensor_view` the attribute indicates that the base address of the `tensor_view` is
divisible by `divisor`. `divisor` must be a positive power of `2`.

The `every` and `along` attributes control which elements are assumed to
satisfy the divisibility property. When splitting the tensor in groups of
size `every` along dimension `along`, the first element of each group is
assumed to satisfy the divisibility property. The other elements are
assumed to be monotonically increasing by `1` within the group. In case
of a `tile` of pointers, the elements are assumed to be monotonically
increasing by the byte width of the pointee type. The size of the last
group may be smaller than `every`.

The `every` and `along` attributes are optional. When missing, they are
assumed to have a default value of `1` and `0` in case of a `tile`.
I.e., all elements of the `tile` are assumed to satisfy the divisibility
property. (The value of `along` does not matter in that case.) If the
predicated value is a `tensor_view` or a 0D `tile`, `every` and `along` cannot be
used.

`every`, and `along` must be used together. If one is specified,
so must be the other.

> **Note**
>
> If the predicated value is a tile of integers, `every` is a property of
> the signed interpretation of the integer values. Otherwise, it is a
> property of the unsigned integer interpretation. E.g., `every = 4`
> is incorrect for the following sequence of “i8” values (written in binary
> form) because they wrap around when interpreted as signed integers:
> `[01111110, 01111111, 10000000, 10000001]`. `every = 2` would
> be correct.

The examples below demonstrate tensors that satisfy the assumed properties.

```mlir
// Example 1: Each pointer is divisible by 16.
// [ 0x10, 0x20, 0x80, 0x10, 0x0, 0x120, ... ]
%0 = cuda_tile.assume #cuda_tile.div_by<16>, %ptrs
    : !cuda_tile.tile<128x!cuda_tile.ptr<f32>>
// Note: Equivalent to #cuda_tile.div_by<16, every 1 along 0>.
```

```mlir
// Example 2: Each integer is divisible by 4.
// [ 16, 24, 8, 4, 12, 12, 0, 16, ... ]
%0 = cuda_tile.assume #cuda_tile.div_by<4>, %t
    : !cuda_tile.tile<128xi32>
```

```mlir
// Example 3: Group size [4].
// [7, 8, 9, 10, 23, 24, 25, 26, 0, 1, 2, 3, ...]
%0 = cuda_tile.assume #cuda_tile.div_by<1, every 4 along 0>, %t
    : !cuda_tile.tile<128xi32>
```

```mlir
// Example 4: 2-d Group size [1, 4] with divisibility 4.
// [ [  4,  5,  6,  7, 12, 13, 14, 15 ],
//   [  8,  9, 10, 11, 24, 25, 26, 27 ],
//   [ 24, 25, 26, 27, 64, 65, 66, 67 ],
//   [  0,  1,  2,  3,  4,  5,  6,  7 ] ]
%0 = cuda_tile.assume #cuda_tile.div_by<4, every 4 along 1>, %t
    : !cuda_tile.tile<4x8xi32>
```

```mlir
// Example 5: 2-d Group size [4, 1] with divisibility 32.
// Note that the elements within each column are monotonically increasing
// by the byte width of the pointee type f32, e.g., 0x20, 0x24, 0x28, 0x2c.
// [ [  0x20, 0x100,  0x40,  0x60,  0x40, 0x200, 0x340,  0x40 ],
//   [  0x24, 0x104,  0x44,  0x64,  0x44, 0x204, 0x344,  0x44 ],
//   [  0x28, 0x108,  0x48,  0x68,  0x48, 0x208, 0x348,  0x48 ],
//   [  0x2c, 0x10c,  0x4c,  0x6c,  0x4c, 0x20c, 0x34c,  0x4c ] ]
%0 = cuda_tile.assume #cuda_tile.div_by<32, every 4 along 0>, %ptrs
    : !cuda_tile.tile<4x8x!cuda_tile.ptr<f32>>
```
