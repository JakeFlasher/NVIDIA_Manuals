---
title: "5.1.2. Alternative Types"
section: "5.1.2"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#alternative-types"
---

### [5.1.2. Alternative Types](https://docs.nvidia.com/cuda/tile-ir/latest/sections#alternative-types)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#alternative-types "Permalink to this headline")

**Tile IR** also supports a set of non-standard but hardware accelerated
floating-point types. Due to the nature of these types and hardware they
each come with a set of restrictions.

| Type | Size | Description |
| --- | --- | --- |
| tf32 | 32 | floating-point format with 8 bits for exponent and 10 bits for mantissa. Storage size is 4 bytes with 4-byte alignment |
| bf16 | 16 | floating-point format with 8 bits for exponent and 7 bits for mantissa |
| e4m3 | 8 | floating-point format with 4 bits for exponent and 3 bits for mantissa |
| e5m2 | 8 | floating-point format with 5 bits for exponent and 2 bits for mantissa |

Tensors of these types may be created, manipulated and loaded and stored
from global memory, but certain computations on them are restricted.
