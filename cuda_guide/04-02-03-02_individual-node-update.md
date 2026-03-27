---
title: "4.2.3.2. Individual Node Update"
section: "4.2.3.2"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html#individual-node-update"
---

### [4.2.3.2. Individual Node Update](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#individual-node-update)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#individual-node-update "Permalink to this headline")

Instantiated graph node parameters can be updated directly. This eliminates the overhead of instantiation as well as the overhead of creating a new `cudaGraph_t`. If the number of nodes requiring update is small relative to the total number of nodes in the graph, it is better to update the nodes individually. The following methods are available for updating `cudaGraphExec_t` nodes:

| API | Node Type |
| --- | --- |
| `cudaGraphExecKernelNodeSetParams()` | Kernel node |
| `cudaGraphExecMemcpyNodeSetParams()` | Memory copy node |
| `cudaGraphExecMemsetNodeSetParams()` | Memory set node |
| `cudaGraphExecHostNodeSetParams()` | Host node |
| `cudaGraphExecChildGraphNodeSetParams()` | Child graph node |
| `cudaGraphExecEventRecordNodeSetEvent()` | Event record node |
| `cudaGraphExecEventWaitNodeSetEvent()` | Event wait node |
| `cudaGraphExecExternalSemaphoresSignalNodeSetParams()` | External semaphore signal node |
| `cudaGraphExecExternalSemaphoresWaitNodeSetParams()` | External semaphore wait node |

Please see the [Graph API](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__GRAPH.html#group__CUDART__GRAPH) for more information on usage and current limitations.
