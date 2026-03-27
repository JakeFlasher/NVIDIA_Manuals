---
title: "Thread ID"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/0t_mma_atom.html#0t_mma_atom--id3"
---

### [Thread ID](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute#id3)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/#id3 "Permalink to this headline")

In the case of Hopper GMMAs, the thread IDs are assigned based on the simple 1D contiguous layout, which makes `thrID` trivial:

```cpp
using ThrID = Layout<_128, _1>;
```
