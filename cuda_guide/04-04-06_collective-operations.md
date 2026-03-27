---
title: "4.4.6. Collective Operations"
section: "4.4.6"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cooperative-groups.html#collective-operations"
---

## [4.4.6. Collective Operations](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#collective-operations)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#collective-operations "Permalink to this headline")

Cooperative Groups includes a set of collective operations that can be performed by a group of threads.
These operations require participation of all threads in the specified group in order to complete the operation.

All threads in the group must pass the same values for corresponding arguments to each collective call, unless different values are explicitly allowed in the [Cooperative Groups API](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/device-callable-apis.html#cg-api-partition-header).
Otherwise the behavior of the call is undefined.
