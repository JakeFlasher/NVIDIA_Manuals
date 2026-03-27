---
title: "Updated Operations"
section: ""
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/release_notes.html#updated-operations"
---

#### [Updated Operations](https://docs.nvidia.com/cuda/tile-ir/latest/sections#updated-operations)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#updated-operations "Permalink to this headline")

- Added `overflow` attribute to `cuda_tile.negi` to control integer overflow behavior.
- Added `rounding_mode` attribute to `cuda_tile.tanh` to control floating-point rounding behavior.
- Added `token` result to `cuda_tile.print_tko` for memory ordering support.
- Added `unsignedCmp` flag to `cuda_tile.for` to support unsigned integer comparison for loop termination.
- Renamed `cuda_tile.print` to `cuda_tile.print_tko` in the textual format. Bytecode encoding is unchanged and remains backward compatible.
