---
title: "Calculating row i given threadblock ID t"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/grouped_scheduler.html#calculating-row-i-given-threadblock-id-t"
---

### [Calculating row i given threadblock ID t](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#calculating-row-i-given-threadblock-id-t)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#calculating-row-i-given-threadblock-id-t "Permalink to this headline")

For a given row i, all threadblock IDs t in that row satisfy the following:

```console
t <= 1 + 2 + 3 + ... + (i-1) + i
```

The closed-form equation for the right-hand side is: `i(i+1)/2`.
Using this, we can solve for `i` given `t`:

```console
t  <= i(i+1)/2
2t <= i^2 + i
2t <= i^2 + i + 0.25 - 0.25
2t + 0.25 <= i^2 + i + 0.25
2t + 0.25 <= (i + 0.5)^2
sqrt(2t + 0.25) - 0.5 <= i
```

To account for fractional values, we set:

```console
i = ceil(sqrt(2t + 0.25) - 0.5)
```

To turn this into a zero-indexed row and work with zero-indexed `t`, we perform:

```console
i = ceil(sqrt(2(t+1) + 0.25) - 0.5) - 1
  = ceil(sqrt(2t + 2.25) - 0.5) - 1
```
