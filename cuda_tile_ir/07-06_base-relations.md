---
title: "7.6. Base relations"
section: "7.6"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/memory_model.html#base-relations"
---

## [7.6. Base relations](https://docs.nvidia.com/cuda/tile-ir/latest/sections#base-relations)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#base-relations "Permalink to this headline")

**Coherence order**

There exists a partial transitive order that relates overlapping write operations, determined at runtime, called _coherence order_.
Two overlapping write operations are related in _coherence order_ if they are _morally strong_ or if they are related in _happens before_ order.

**Program order**

A memory operation \(\(A\)\) is program order before a memory operation \(\(B\)\) if the instruction that gave rise to \(\(A\)\) is before the instruction that gave rise to \(\(B\)\) in the program source.

**Waits-for order**

An operation \(\(A\)\) _waits-for_ an operation \(\(B\)\) if:
- an instruction \(\(I_A\)\) gave rise to \(\(A\)\), \(\(I_A\)\) produced a token \(\(t\)\), an instruction \(\(I_B\)\) gave rise to \(\(B\)\) and \(\(I_B\)\) depends upon the token \(\(t\)\); or
- there is some operation \(\(C\)\) such that \(\(A\)\) _waits-for_ \(\(C\)\) and \(\(C\)\) _waits-for_ \(\(B\)\).

**Reads from**

An read operation \(\(R\)\) _reads-from_ a write operation \(\(W\)\) when \(\(R\)\) and \(\(W\)\) access the same location and \(\(R\)\) reads the value written by \(\(W\)\).

**Read-modify-write order**

Read-modify-write atomics generate a pair of memory operations for each element location within a tile.
Each pair of memory operations is related by _read-modify-write order_.

**Atomicity**

When an atomic operation \(\(A\)\) and a write \(\(W\)\) overlap and are _morally strong_, then the following two communications cannot both exist in the same execution:

- \(\(A\)\) reads from a write \(\(W'\)\) that precedes \(\(W\)\) in _coherence order_.
- \(\(A\)\) follows \(\(W\)\) in _coherence order_.
