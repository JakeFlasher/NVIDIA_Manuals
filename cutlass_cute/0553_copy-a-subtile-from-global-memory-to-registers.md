---
title: "Copy a subtile from global memory to registers"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/03_tensor.html#copy-a-subtile-from-global-memory-to-registers"
---

### [Copy a subtile from global memory to registers](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute#copy-a-subtile-from-global-memory-to-registers)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/#copy-a-subtile-from-global-memory-to-registers "Permalink to this headline")

The following example copies rows of a matrix (with any `Layout`)
from global memory to register memory,
then executes some algorithm `do_something`
on the row that lives in register memory.

```c++
Tensor gmem = make_tensor(ptr, make_shape(Int<8>{}, 16));  // (_8,16)
Tensor rmem = make_tensor_like(gmem(_, 0));                // (_8)
for (int j = 0; j < size<1>(gmem); ++j) {
  copy(gmem(_, j), rmem);
  do_something(rmem);
}
```

This code does not need to know anything about the `Layout` of `gmem`
other than that it is rank-2 and that the first mode has a static size.
The following code checks both of those conditions at compile time.

```c++
CUTE_STATIC_ASSERT_V(rank(gmem) == Int<2>{});
CUTE_STATIC_ASSERT_V(is_static<decltype(shape<0>(gmem))>{});
```

Extending this example using the tiling utilities detailed in [the `Layout` algebra section](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/02_layout_algebra.html), we can copy an arbitrary subtile of a tensor using almost the same code as above.

```c++
Tensor gmem = make_tensor(ptr, make_shape(24, 16));         // (24,16)

auto tiler         = Shape<_8,_4>{};                        // 8x4 tiler
//auto tiler       = Tile<Layout<_8,_3>, Layout<_4,_2>>{};  // 8x4 tiler with stride-3 and stride-2
Tensor gmem_tiled  = zipped_divide(gmem, tiler);            // ((_8,_4),Rest)
Tensor rmem        = make_tensor_like(gmem_tiled(_, 0));    // ((_8,_4))
for (int j = 0; j < size<1>(gmem_tiled); ++j) {
  copy(gmem_tiled(_, j), rmem);
  do_something(rmem);
}
```

This applies a statically shaped `Tiler` to the global memory `Tensor`, creates a register `Tensor` that is compatible with the shape of that tile, then loops through each tile to copy it into memory and `do_something`.
