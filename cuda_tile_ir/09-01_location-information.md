---
title: "9.1. Location Information"
section: "9.1"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/debug_info.html#location-information"
---

## [9.1. Location Information](https://docs.nvidia.com/cuda/tile-ir/latest/sections#location-information)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#location-information "Permalink to this headline")

**Tile IR** requires that scope metadata is attached to all operations within a **Tile IR** module to generate debug info of any kind. Scope metadata is required in addition to simple file, line, column location information. Providing simple file, line, column location information alone without scope metadata results in a very poor user experience. Scoped metadata is required to generate DWARF information and DWARF information is required by all developer tools including debuggers and profilers.

Example of a simple file, line, column location: `loc("/tmp/foo.py":1:4)`

Example of a location fused with scope metadata: `#cuda_tile.di_loc<loc("/tmp/foo.py":1:4) in #subprogram`
