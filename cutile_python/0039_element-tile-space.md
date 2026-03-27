---
title: "Element & Tile Space"
section: ""
source: "https://docs.nvidia.com/cuda/cutile-python/data.html#element-tile-space"
---

## [Element & Tile Space](https://docs.nvidia.com/cuda/cutile-python#element-tile-space)[](https://docs.nvidia.com/cuda/cutile-python/#element-tile-space "Permalink to this headline")

![_images/cutile__indexing__array_shape_12x16__tile_shape_2x4__tile_grid_6x4__dark_background.svg](images/_______-____-______1.svg)
![_images/cutile__indexing__array_shape_12x16__tile_shape_2x4__tile_grid_6x4__light_background.svg](images/_______-____-______2.svg)
![_images/cutile__indexing__array_shape_12x16__tile_shape_4x2__tile_grid_3x8__dark_background.svg](images/_______-____-______3.svg)
![_images/cutile__indexing__array_shape_12x16__tile_shape_4x2__tile_grid_3x8__light_background.svg](images/_______-____-______4.svg)
The _element space_ of an array is the multidimensional space of elements contained in that array,
stored in memory according to a certain layout (row major, column major, etc).

The _tile space_ of an array is the multidimensional space of tiles into that array of a certain
tile shape.
A tile index `(i, j, ...)` with shape `S` refers to the elements of the array that belong to the
`(i+1)`-th, `(j+1)`-th, … tile.

When accessing the elements of an array using tile indices, the multidimensional memory layout of the array is used.
To access the tile space with a different memory layout, use the *order* parameter of load/store operations.
