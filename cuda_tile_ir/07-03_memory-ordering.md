---
title: "7.3. Memory ordering"
section: "7.3"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/memory_model.html#memory-ordering"
---

## [7.3. Memory ordering](https://docs.nvidia.com/cuda/tile-ir/latest/sections#memory-ordering)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#memory-ordering "Permalink to this headline")

Memory operations have a memory ordering parameter which controls how that operation can be used for synchronization.
Synchronization through memory is a two-party process, which requires a releaser and an acquirer observing the same location.
When a pair of memory accesses synchronize through memory it establishes a _happens before_ relationship.
See the happens before definition below.

Any ordering other than `weak` requires a scope to be set.

| Memory ordering | Description |
| --- | --- |
| `weak` | No concurrent accesses to the source/destination location. |
| `relaxed` | There may be concurrent access to the location, but this access does not establish a happens-before relationship. |
| `release` | There may be concurrent access to the location. If this release is observed with an acquire operation, then happens before is established. |
| `acquire` | There may be concurrent accesses to the location. If this acquire observes a release operation, then happens before is established. |
| `acq_rel` | There may be concurrent accesses to the location. This has the effect of both a release and acquire operation. |
