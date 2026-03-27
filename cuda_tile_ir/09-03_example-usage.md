---
title: "9.3. Example Usage"
section: "9.3"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/debug_info.html#example-usage"
---

## [9.3. Example Usage](https://docs.nvidia.com/cuda/tile-ir/latest/sections#example-usage)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#example-usage "Permalink to this headline")

Below is an example showing how to use **Tile IR** debug info:

```mlir
#file= #cuda_tile.di_file<"foo.py" in "/tmp/">

#compile_unit= #cuda_tile.di_compile_unit<
  file = #file
>

#subprogram= #cuda_tile.di_subprogram<
  file = #file,
  line = 1,
  name = "test_kernel",
  linkageName = "kernel",
  compileUnit = #compile_unit,
  scopeLine = 1
>

#block= #cuda_tile.di_lexical_block<
  scope = #subprogram,
  file = #file,
  line = 1,
  column = 4
>

#loc_fn= #cuda_tile.di_loc<loc("/tmp/foo.py":1:4) in #subprogram>
#loc_return= #cuda_tile.di_loc<loc("/tmp/foo.py":5:4) in #block>

cuda_tile.module @kernels attributes { } {
  entry @kernel() {
    return loc(#loc_return)
  } loc(#loc_fn)
}
```
