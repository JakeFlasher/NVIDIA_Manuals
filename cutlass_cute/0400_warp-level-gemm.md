---
title: "Warp-level GEMM"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html#warp-level-gemm"
---

### [Warp-level GEMM](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#warp-level-gemm)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#warp-level-gemm "Permalink to this headline")

The warp-level GEMM maps to the warp-level parallelism within the CUDA execution model. Multiple
warps within a threadblock fetch data from shared memory into registers and perform computations.
Warp-level GEMMs may be implemented either by TensorCores issuing
[mma.sync](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-mma)
or [wmma](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-wmma-mma)
instructions, or by thread-level matrix computations issued to CUDA cores.
For maximum performance, access to shared memory should be bank conflict free. To maximize data
reuse within the warp, a large warp-level GEMM tile should be chosen.
