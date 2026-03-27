---
title: "2.5. Cross TileBlock Communication"
section: "2.5"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/prog_model.html#cross-tileblock-communication"
---

## [2.5. Cross TileBlock Communication](https://docs.nvidia.com/cuda/tile-ir/latest/sections#cross-tileblock-communication)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#cross-tileblock-communication "Permalink to this headline")

```mlir
  cuda_tile.module @hello_cross_block {
    global @_global_printf_mutex <i32: 1> : tile<1xi32>

    entry @hello_cross_block_kernel() {
      %idx, %idy, %idz = get_tile_block_id : tile<i32>
      %tilex, %tiley, %tilez = get_num_tile_blocks : tile<i32>
      %2 = get_global @_global_printf_mutex : tile<ptr<i32>>
      %3 = cuda_tile.constant <i32: 0> : tile<i32>
      %4 = cuda_tile.constant <i32: 1> : tile<i32>
      loop {
        %t1 = make_token : token
        %6, %t2 = atomic_cas_tko relaxed device %2, %4, %3 token=%t1: tile<!cuda_tile.ptr<i32>>, tile<i32> -> tile<i32>, !cuda_tile.token
        %7 = trunci %6 : tile<i32> -> tile<i1>
        if %7 {
          break
        }
      }
      print "current tile: %i / %i\0A", %idx, %tilex : tile<i32>, tile<i32>
      %5, %t4 = atomic_rmw_tko relaxed device %2, xchg, %4: tile<ptr<i32>>, tile<i32> -> tile<i32>, !cuda_tile.token
      return
    }
  }
```
