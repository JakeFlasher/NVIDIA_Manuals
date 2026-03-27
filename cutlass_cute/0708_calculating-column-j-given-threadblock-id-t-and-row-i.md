---
title: "Calculating column j given threadblock ID t and row i"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/grouped_scheduler.html#calculating-column-j-given-threadblock-id-t-and-row-i"
---

### [Calculating column j given threadblock ID t and row i](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#calculating-column-j-given-threadblock-id-t-and-row-i)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#calculating-column-j-given-threadblock-id-t-and-row-i "Permalink to this headline")

For a given row `i`, all threadblock IDs `t` in that row also satisfy the following:

```console
    t > 1 + 2 + 3 + ... + (i-2) + (i-1)
--> t > i(i-1)/2
```

Threadblock IDs within a given row are sequential, so the one-indexed column ID
for one-indexed threadblock ID `t` and row `i` is:

```console
j = t - (i(i-1)/2)
```

The zero-indexed version becomes:

```console
j = (t+1) - (i(i+1)/2) -1
  = t - (i(i+1)/2)
```
