---
title: "8.2.3. Explicit Overflow Annotations"
section: "8.2.3"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html#explicit-overflow-annotations"
---

### [8.2.3. Explicit Overflow Annotations](https://docs.nvidia.com/cuda/tile-ir/latest/sections#explicit-overflow-annotations)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#explicit-overflow-annotations "Permalink to this headline")

Some operations such as [cuda_tile.addi](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#op-cuda-tile-addi) support an explicit overflow annotation that expresses the expected overflow behavior
of the operation.

These attributes serve as assumptions that an implementation may use to reason about the operation. It is the responsibility of the code generator
to ensure that the operation respects these assumptions dynamically during execution.

We recommend that generators of **Tile IR** programs utilize these annotations to help the implementation reason about the overflow behavior of the
operation, enabling extra optimization opportunities.
