---
title: "Contents"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/0t_mma_atom.html#0t_mma_atom--id2"
---

#### [Contents](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute#id2)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/#id2 "Permalink to this headline")

An `MMA_Traits` specialization defines the following public type aliases.

- `ValTypeD`: Logical compute type of the D matrix
- `ValTypeA`: Logical compute type of the A matrix
- `ValTypeB`: Logical compute type of the B matrix
- `ValTypeC`: Logical compute type of the C matrix
- `Shape_MNK`: Logical MxNxK shape of the MMA operation
- `ThrID`: Logical thread mapping within the single MMA operation
(specifying the thread, quadpair, warp, or warpgroup view)
- `ALayout`: Mapping of (thread,value) pairs to coordinates in the MxK A matrix
- `BLayout`: Mapping of (thread,value) pairs to coordinates in the NxK B matrix
- `CLayout`: Mapping of (thread,value) pairs to coordinates in the MxN C matrix
