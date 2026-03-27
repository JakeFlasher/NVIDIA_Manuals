---
title: "4.2.7. Using Graph APIs"
section: "4.2.7"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html#using-graph-apis"
---

## [4.2.7. Using Graph APIs](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#using-graph-apis)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#using-graph-apis "Permalink to this headline")

`cudaGraph_t` objects are not thread-safe. It is the responsibility of the user to ensure that multiple threads do not concurrently access the same `cudaGraph_t`.

A `cudaGraphExec_t` cannot run concurrently with itself. A launch of a `cudaGraphExec_t` will be ordered after previous launches of the same executable graph.

Graph execution is done in streams for ordering with other asynchronous work. However, the stream is for ordering only; it does not constrain the internal parallelism of the graph, nor does it affect where graph nodes execute.

See [Graph API.](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__GRAPH.html#group__CUDART__GRAPH)
