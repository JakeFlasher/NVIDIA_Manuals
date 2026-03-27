---
title: "4.3.4.4.4. Disabling Reuse Policies"
section: "4.3.4.4.4"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/stream-ordered-memory-allocation.html#disabling-reuse-policies"
---

#### [4.3.4.4.4. Disabling Reuse Policies](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#disabling-reuse-policies)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#disabling-reuse-policies "Permalink to this headline")

While the controllable reuse policies improve memory reuse, users may want to
disable them. Allowing opportunistic reuse (such as
`cudaMemPoolReuseAllowOpportunistic`) introduces run to run variance in
allocation patterns based on the interleaving of CPU and GPU execution.
Internal dependency insertion (such as
`cudaMemPoolReuseAllowInternalDependencies`) can serialize work in
unexpected and potentially non-deterministic ways when the user would rather
explicitly synchronize an event or stream on allocation failure.
