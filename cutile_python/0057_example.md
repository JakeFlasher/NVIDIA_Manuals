---
title: "Example"
section: ""
source: "https://docs.nvidia.com/cuda/cutile-python/performance.html#example"
---

### [Example](https://docs.nvidia.com/cuda/cutile-python#example)[](https://docs.nvidia.com/cuda/cutile-python/#example "Permalink to this headline")

```default
import cuda.tile as ct

TILE_SIZE = 16

@ct.kernel
def load_store_with_hints_kernel(x, y):
    bid = ct.bid(0)
    tx = ct.load(
        x,
        index=(bid,),
        shape=(TILE_SIZE,),
        latency=8,        # high-latency DRAM load
    )
    ct.store(
        y,
        index=(bid,),
        tile=tx,
        latency=2,        # cheaper write
        allow_tma=False,  # disallow TMA
    )
```
