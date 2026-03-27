---
title: "Improving Load Balance by Sorting Problems"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/grouped_scheduler.html#improving-load-balance-by-sorting-problems"
---

# [Improving Load Balance by Sorting Problems](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#improving-load-balance-by-sorting-problems)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#improving-load-balance-by-sorting-problems "Permalink to this headline")

The grouped kernel schedulers assign a nearly equal number
of tiles to each block participating in the grouped kernel. Every tile in the
group has the same M and N dimensions. However, the K dimension of each
tile depends on the K dimension of the problem, so tiles may have different
K dimensions. Thus, the K dimension of the
tile plays a significant role in determining how long it takes for a given
tile to be computed.
