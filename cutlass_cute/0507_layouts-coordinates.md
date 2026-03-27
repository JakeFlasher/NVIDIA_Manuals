---
title: "Layouts Coordinates"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/01_layout.html#layouts-coordinates"
---

### [Layouts Coordinates](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute#layouts-coordinates)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/#layouts-coordinates "Permalink to this headline")

With the notion of compatibility above, we emphasize that every `Layout` accepts multiple kinds of coordinates. Every `Layout` accepts coordinates for any `Shape` that is compatible with it. CuTe provides mappings between these sets of coordinates via a colexicographical order.

Thus, all Layouts provide two fundamental mappings:

- the map from an input coordinate to the corresponding natural coordinate via the `Shape`, and
- the map from a natural coordinate to the index via the `Stride`.
