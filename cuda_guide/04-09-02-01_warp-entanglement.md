---
title: "4.9.2.1. Warp Entanglement"
section: "4.9.2.1"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-barriers.html#warp-entanglement"
---

### [4.9.2.1. Warp Entanglement](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#warp-entanglement)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#warp-entanglement "Permalink to this headline")

Warp-divergence affects the number of times an arrive on operation updates the barrier. If the invoking warp is fully converged, then the barrier is updated once. If the invoking warp is fully diverged, then 32 individual updates are applied to the barrier.

> **Note**
>
> It is recommended that `arrive-on(bar)` invocations are used by converged threads to minimize updates to the barrier object. When code preceding these operations diverges threads, then the warp should be re-converged, via `__syncwarp` before invoking arrive-on operations.
