---
title: "9.1.4. Trade-offs"
section: "9.1.4"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/debug_info.html#trade-offs"
---

### [9.1.4. Trade-offs](https://docs.nvidia.com/cuda/tile-ir/latest/sections#trade-offs)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#trade-offs "Permalink to this headline")

When deciding between the two options above, the **Tile IR** producer should consider the following trade-offs:

1. Providing location information with scope metadata may be harder for the **Tile IR** producer to implement, however, the **Tile IR** producer retains full control to describe its source code to the compiler which may result in a smoother debugging experience for the end user.
2. Providing simple file line, location information is very simple for the **Tile IR** producer to implement but cedes control to the pass which may not be able to accurately synthesize debug info for all cases possibly resulting in a degraded debugging experience for the end user. E.g. the pass may be able to synthesize function name, guess at linkage name, but not do a very good job of getting the line or column number of the function.
