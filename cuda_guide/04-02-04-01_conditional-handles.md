---
title: "4.2.4.1. Conditional Handles"
section: "4.2.4.1"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html#conditional-handles"
---

### [4.2.4.1. Conditional Handles](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#conditional-handles)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#conditional-handles "Permalink to this headline")

A condition value is represented by `cudaGraphConditionalHandle` and is created by `cudaGraphConditionalHandleCreate()`.

The handle must be associated with a single conditional node. Handles cannot be destroyed and as such there is no need to keep track of them.

If `cudaGraphCondAssignDefault` is specified when the handle is created, the condition value will be initialized to the specified default at the beginning of each graph execution. If this flag is not provided, the condition value is undefined at the start of each graph execution and code should not assume that the condition value persists across executions.

The default value and flags associated with a handle will be updated during [whole graph update](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#cuda-graphs-whole-graph-update).
