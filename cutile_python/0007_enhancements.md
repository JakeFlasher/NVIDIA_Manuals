---
title: "Enhancements"
section: ""
source: "https://docs.nvidia.com/cuda/cutile-python/generated/release_notes.html#enhancements"
---

### [Enhancements](https://docs.nvidia.com/cuda/cutile-python/generated#enhancements)[](https://docs.nvidia.com/cuda/cutile-python/generated/#enhancements "Permalink to this headline")

- Erase the distinction between scalars and zero-dimensional tiles.
They are now completely interchangeable.
- `~x` for const boolean `x` will raise a TypeError to prevent inconsistent
results compared to `~x` on a boolean Tile.
- Add `TileUnsupportedFeatureError` to the public API.
