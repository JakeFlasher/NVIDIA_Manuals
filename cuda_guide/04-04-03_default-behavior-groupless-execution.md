---
title: "4.4.3. Default Behavior / Groupless Execution"
section: "4.4.3"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cooperative-groups.html#default-behavior-groupless-execution"
---

## [4.4.3. Default Behavior / Groupless Execution](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#default-behavior-groupless-execution)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#default-behavior-groupless-execution "Permalink to this headline")

Groups representing the grid and thread blocks are implicitly created based on the kernel launch configuration.
These “implicit” groups provide a starting point that developers can explicitly decompose into finer grained groups.
Implicit groups can be accessed using the following methods:

| Accessor | Group Scope |
| --- | --- |
| `this_thread_block()` | Returns the handle to a group containing all threads in current thread block. |
| `this_grid()` | Returns the handle to a group containing all threads in the grid. |
| `coalesced_threads()` [^[1]] | Returns the handle to a group of currently active threads in a warp. |
| `this_cluster()` [^[2]] | Returns the handle to a group of threads in the current cluster. |

[^[1]]: The `coalesced_threads()` operator returns the set of active threads at that point in time, and makes no guarantee about which threads are returned (as long as they are active) or that they will stay coalesced throughout execution.

[^[2]]: The `this_cluster()` assumes a 1x1x1 cluster when a non-cluster grid is launched. Requires Compute Capability 9.0 or greater.

More information is available in the [Cooperative Groups API](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/device-callable-apis.html#cg-api-common-header).
