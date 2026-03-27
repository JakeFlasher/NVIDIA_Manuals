---
title: "Load/Store"
section: ""
source: "https://docs.nvidia.com/cuda/cutile-python/operations.html#load-store"
---

## [Load/Store](https://docs.nvidia.com/cuda/cutile-python#load-store)[](https://docs.nvidia.com/cuda/cutile-python/#load-store "Permalink to this headline")

| [`bid`](https://docs.nvidia.com/cuda/cutile-python/generated/cuda.tile.bid.html#cuda.tile.bid "cuda.tile.bid") | Gets the index of current block. |
| --- | --- |
| [`num_blocks`](https://docs.nvidia.com/cuda/cutile-python/generated/cuda.tile.num_blocks.html#cuda.tile.num_blocks "cuda.tile.num_blocks") | Gets the number of blocks along the axis. |
| [`num_tiles`](https://docs.nvidia.com/cuda/cutile-python/generated/cuda.tile.num_tiles.html#cuda.tile.num_tiles "cuda.tile.num_tiles") | Gets the number of tiles in the [tile space](https://docs.nvidia.com/cuda/cutile-python/data.html#data-element-tile-space) of the array along the *axis*. |
| [`load`](https://docs.nvidia.com/cuda/cutile-python/generated/cuda.tile.load.html#cuda.tile.load "cuda.tile.load") | Loads a tile from the *array* which is partitioned into a [tile space](https://docs.nvidia.com/cuda/cutile-python/data.html#data-element-tile-space). |
| [`store`](https://docs.nvidia.com/cuda/cutile-python/generated/cuda.tile.store.html#cuda.tile.store "cuda.tile.store") | Stores a *tile* value into the *array* at the *index* of its [tile space](https://docs.nvidia.com/cuda/cutile-python/data.html#data-element-tile-space). |
| [`gather`](https://docs.nvidia.com/cuda/cutile-python/generated/cuda.tile.gather.html#cuda.tile.gather "cuda.tile.gather") | Loads a tile from the *array* elements specified by *indices*. |
| [`scatter`](https://docs.nvidia.com/cuda/cutile-python/generated/cuda.tile.scatter.html#cuda.tile.scatter "cuda.tile.scatter") | Stores a tile *value* into the *array* elements specified by *indices*. |
