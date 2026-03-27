---
title: "8.8.1. Integer Arithmetic"
section: "8.8.1"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html#integer-arithmetic"
---

### [8.8.1. Integer Arithmetic](https://docs.nvidia.com/cuda/tile-ir/latest/sections#integer-arithmetic)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#integer-arithmetic "Permalink to this headline")

Integer types in **Tile IR** are signless, which is importantly not the same as unsigned. We store all integers in a two’s complement representation
and with required operations supporting a *signed* or *unsigned* flag as needed. This design allows us to not have to differentiate between signed
and unsigned integer types at the IR level and keeps sign information local to the operation.

For the `i1` type, unsigned operations see values 0/1, while signed operations see values 0/-1,  with all i1 values canonicalized to 0x00 (false) or 0x01 (true)
for consistent LSB-only semantics.
