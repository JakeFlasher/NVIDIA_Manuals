---
title: "4.10.4. Warp Entanglement"
section: "4.10.4"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/pipelines.html#pipelines--warp-entanglement"
---

## [4.10.4. Warp Entanglement](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#warp-entanglement)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#warp-entanglement "Permalink to this headline")

The pipeline mechanism is shared among CUDA threads in the same warp. This sharing causes sequences of submitted operations to be entangled within a warp, which can impact performance under certain circumstances.

**Commit**. The commit operation is coalesced such that the pipeline’s sequence is incremented once for all converged threads that invoke the commit operation and their submitted operations are batched together. If the warp is fully converged, the sequence is incremented by one and all submitted operations will be batched in the same stage of the pipeline; if the warp is fully diverged, the sequence is incremented by 32 and all submitted operations will be spread to different stages.

- Let _PB_ be the warp-shared pipeline’s _actual_ sequence of operations.

`PB = {BP0, BP1, BP2, …, BPL}`
- Let _TB_ be a thread’s _perceived_ sequence of operations, as if the sequence were only incremented by this thread’s invocation of the commit operation.

`TB = {BT0, BT1, BT2, …, BTL}`

> The `pipeline::producer_commit()` return value is from the thread’s _perceived_ batch sequence.

- An index in a thread’s perceived sequence always aligns to an equal or larger index in the actual warp-shared sequence. The sequences are equal only when all commit operations are invoked from fully converged threads.

`BTn ≡ BPm` where `n <= m`

For example, when a warp is fully diverged:

- The warp-shared pipeline’s actual sequence would be: `PB = {0, 1, 2, 3, ..., 31}` (`PL=31`).
- The perceived sequence for each thread of this warp would be:
  - Thread 0: `TB = {0}` (`TL=0`)
  - Thread 1: `TB = {0}` (`TL=0`)
  - `…`
  - Thread 31: `TB = {0}` (`TL=0`)

**Wait**. A CUDA thread invokes `pipeline::consumer_wait()` or `pipeline_consumer_wait_prior<N>()` to wait for batches in the _perceived_ sequence `TB` to complete. Note that `pipeline::consumer_wait()` is equivalent to `pipeline_consumer_wait_prior<N>()`, where `N = PL`.

The _wait prior_ variants wait for batches in the _actual_ sequence at least up to and including `PL-N`. Since `TL <= PL`, waiting for batch up to and including `PL-N` includes waiting for batch `TL-N`. Thus, when `TL < PL`, the thread will unintentionally wait for additional, more recent batches. In the extreme fully-diverged warp example above, each thread could wait for all 32 batches.

> **Note**
>
> It is recommended that commit invocations are by converged threads to not over-wait, by keeping threads’ perceived sequence of batches aligned with the actual sequence.
>
> When code preceding these operations diverges threads, then the warp should be re-converged, via `__syncwarp` before invoking commit operations.
