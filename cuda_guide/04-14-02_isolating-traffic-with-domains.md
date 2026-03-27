---
title: "4.14.2. Isolating Traffic with Domains"
section: "4.14.2"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/memory-sync-domains.html#isolating-traffic-with-domains"
---

## [4.14.2. Isolating Traffic with Domains](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#isolating-traffic-with-domains)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#isolating-traffic-with-domains "Permalink to this headline")

Beginning with compute capability 9.0 (Hopper architecture) GPUs and CUDA 12.0, the memory synchronization domains feature provides a way to alleviate such interference. In exchange for explicit assistance from code, the GPU can reduce the net cast by a fence operation. Each kernel launch is given a domain ID. Writes and fences are tagged with the ID, and a fence will only order writes matching the fence’s domain. In the concurrent compute vs communication example, the communication kernels can be placed in a different domain.

When using domains, code must abide by the rule that **ordering or synchronization between distinct domains on the same GPU requires system-scope fencing**. Within a domain, device-scope fencing remains sufficient. This is necessary for cumulativity as one kernel’s writes will not be encompassed by a fence issued from a kernel in another domain. In essence, cumulativity is satisfied by ensuring that cross-domain traffic is flushed to the system scope ahead of time.

Note that this modifies the definition of `thread_scope_device`. However, because kernels will default to domain 0 as described below, backward compatibility is maintained.
