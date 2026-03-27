---
title: "4.2.3.3. Individual Node Enable"
section: "4.2.3.3"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html#individual-node-enable"
---

### [4.2.3.3. Individual Node Enable](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#individual-node-enable)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#individual-node-enable "Permalink to this headline")

Kernel, memset and memcpy nodes in an instantiated graph can be enabled or disabled using the `cudaGraphNodeSetEnabled()` API. This allows the creation of a graph which contains a superset of the desired functionality which can be customized for each launch. The enable state of a node can be queried using the `cudaGraphNodeGetEnabled()` API.

A disabled node is functionally equivalent to empty node until it is re-enabled. Node parameters are not affected by enabling/disabling a node. Enable state is unaffected by individual node update or whole graph update with `cudaGraphExecUpdate()`. Parameter updates while the node is disabled will take effect when the node is re-enabled.

Refer to the [Graph API](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__GRAPH.html#group__CUDART__GRAPH) for more information on usage and current limitations.
