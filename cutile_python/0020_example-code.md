---
title: "Example Code"
section: ""
source: "https://docs.nvidia.com/cuda/cutile-python/quickstart.html#example-code"
---

## [Example Code](https://docs.nvidia.com/cuda/cutile-python#example-code)[](https://docs.nvidia.com/cuda/cutile-python/#example-code "Permalink to this headline")

The following example shows vector addition, a typical first kernel for CUDA, but uses cuTile for tile-based programming. This makes use of a 1-dimensional tile to add two 1-dimensional vectors.

This example shows a structure common to cuTile kernels:

- Load one or more tiles from GPU memory
- Perform computation(s) on the tile(s), resulting in new tile(s)
- Write the resulting tile(s) out to GPU memory

In this case, the kernel loads tiles from two vectors, `a` and `b`. These loads create tiles called `a_tile` and `b_tile`. These tiles are added together to form a third tile, called `result`. In the last step, the kernel stores the `result` tile to the output vector `c`.
More samples can be found in the cuTile Python [repository](https://github.com/nvidia/cutile-python).

```python
# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""
Example demonstrating simple vector addition.
Shows how to perform elementwise operations on vectors.
"""

import cupy as cp
import numpy as np
import cuda.tile as ct

@ct.kernel
def vector_add(a, b, c, tile_size: ct.Constant[int]):
    # Get the 1D pid
    pid = ct.bid(0)

    # Load input tiles
    a_tile = ct.load(a, index=(pid,), shape=(tile_size,))
    b_tile = ct.load(b, index=(pid,), shape=(tile_size,))

    # Perform elementwise addition
    result = a_tile + b_tile

    # Store result
    ct.store(c, index=(pid, ), tile=result)

def test():
    # Create input data
    vector_size = 2**12
    tile_size = 2**4
    grid = (ct.cdiv(vector_size, tile_size), 1, 1)

    rng = cp.random.default_rng()
    a = rng.random(vector_size)
    b = rng.random(vector_size)
    c = cp.zeros_like(a)

    # Launch kernel
    ct.launch(cp.cuda.get_current_stream(),
              grid,  # 1D grid of processors
              vector_add,
              (a, b, c, tile_size))

    # Copy to host only to compare
    a_np = cp.asnumpy(a)
    b_np = cp.asnumpy(b)
    c_np = cp.asnumpy(c)

    # Verify results
    expected = a_np + b_np
    np.testing.assert_array_almost_equal(c_np, expected)

    print("✓ vector_add_example passed!")

if __name__ == "__main__":
    test()
```

Run this from a command line as shown below. If everything has been setup correctly, the test will print that the example passed.

```bash
$ python3 samples/quickstart/VectorAdd_quickstart.py
✓ vector_add_example passed!
```

To run more of the cuTile Python examples, you can directly run the samples by invoking them in the same way as the quickstart example:

```bash
$ python3 samples/FFT.py
# output not shown
```

You can also use pytest to run all the samples:

```bash
$  pytest samples
========================= test session starts =========================
platform linux -- Python 3.12.3, pytest-9.0.1, pluggy-1.6.0
rootdir: /home/ascudiero/sw/cutile-python
configfile: pytest.ini
collected 6 items

samples/test_samples.py ......                                  [100%]

========================= 6 passed in 30.74s ==========================
```
