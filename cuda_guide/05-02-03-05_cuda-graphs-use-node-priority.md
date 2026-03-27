---
title: "5.2.3.5. CUDA_GRAPHS_USE_NODE_PRIORITY"
section: "5.2.3.5"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/environment-variables.html#cuda-graphs-use-node-priority"
---

### [5.2.3.5. CUDA_GRAPHS_USE_NODE_PRIORITY](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#cuda-graphs-use-node-priority)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#cuda-graphs-use-node-priority "Permalink to this headline")

The environment variable controls the CUDA graph’s execution priority relative to the stream priority it inherits from the stream in which it is launched.

`CUDA_GRAPHS_USE_NODE_PRIORITY` overrides the [cudaGraphInstantiateFlagUseNodePriority](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__GRAPH.html#group__CUDART__GRAPH_1gd4d586536547040944c05249ee26bc62) flag on graph instantiation.

**Possible Values**:

- `0`: Inherit the priority of the stream the graph is launched into (default).
- `1`: Honor per-node launch priorities. The CUDA runtime treats node-level priorities as a scheduling hint for ready-to-run graph nodes.

**Example**:

```bash
CUDA_GRAPHS_USE_NODE_PRIORITY=1
```
