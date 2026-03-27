---
title: "11.1.1. Hello Tile Block"
section: "11.1.1"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/appendix.html#hello-tile-block"
---

### [11.1.1. Hello Tile Block](https://docs.nvidia.com/cuda/tile-ir/latest/sections#hello-tile-block)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#hello-tile-block "Permalink to this headline")

```mlir
cuda_tile.module @hello_world_module {
    entry @hello_world_kernel() {
        print "Hello World!\n"
    }
}
```
