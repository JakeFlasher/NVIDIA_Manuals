---
title: "4.2.1. Magic Number"
section: "4.2.1"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/bytecode.html#magic-number"
---

### [4.2.1. Magic Number](https://docs.nvidia.com/cuda/tile-ir/latest/sections#magic-number)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#magic-number "Permalink to this headline")

The **Tile IR** bytecode magic number consumes 8 bytes and is `\x7FTileIR\x00`. The magic number
must be present at the beginning of the bytecode file to be accepted by the driver.
