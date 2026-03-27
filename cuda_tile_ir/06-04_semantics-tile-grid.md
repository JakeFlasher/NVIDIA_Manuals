---
title: "6.4. Tile Grid"
section: "6.4"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/semantics.html#semantics--tile-grid"
---

## [6.4. Tile Grid](https://docs.nvidia.com/cuda/tile-ir/latest/sections#tile-grid)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#tile-grid "Permalink to this headline")

During execution, the abstract machine instantiates a grid of tile blocks. A tile grid is grid of
tile blocks arranged in a 1-, 2-, or 3-dimensional array.

Each position of the grid corresponds to a single independent tile kernel instance.

The abstract machine stores a sequence of tile blocks, \(\(TB\)\), that are indexed by a grid of coordinates, \(\(\vec{g}\)\).
Each tile block is assigned unique tile block id based on the grid size and the tile block’s position within the grid.

The bijective mapping between 1-, 2-, or 3-dimensional grid coordinates and flattened tile block ids is
computed as follows:

$$
\[\begin{split}\begin{aligned}
\text{tile\_grid}(\vec{i}) &=
\begin{cases}
i_0 & \text{where grid} = (x) \\
i_0 \cdot x + i_1 & \text{where grid} = (x,y) \\
i_0 \cdot x + i_1 \cdot y + i_2 & \text{where grid} = (x,y,z)
\end{cases} \\
&\text{where } \vec{i} = (i_0, \ldots, i_n) \\
&\text{and } 0 \leq i_k < \text{grid}_k \text{ for all } k \leq 3
\end{aligned}\end{split}\]
$$
