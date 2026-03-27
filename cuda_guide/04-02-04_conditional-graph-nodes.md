---
title: "4.2.4. Conditional Graph Nodes"
section: "4.2.4"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html#conditional-graph-nodes"
---

## [4.2.4. Conditional Graph Nodes](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#conditional-graph-nodes)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#conditional-graph-nodes "Permalink to this headline")

Conditional nodes allow conditional execution and looping of a graph contained within the conditional node. This allows dynamic and iterative workflows to be represented completely within a graph and frees up the host CPU to perform other work in parallel.

Evaluation of the condition value is performed on the device when the dependencies of the conditional node have been met. Conditional nodes can be one of the following types:

- Conditional [IF nodes](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#cuda-graphs-conditional-if-nodes) execute their body graph once if the condition value is non-zero when the node is executed. An optional second body graph can be provided and this will be executed once if the condition value is zero when the node is executed.
- Conditional [WHILE nodes](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#cuda-graphs-conditional-while-nodes) execute their body graph if the condition value is non-zero when the node is executed and will continue to execute their body graph until the condition value is zero.
- Conditional [SWITCH nodes](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#cuda-graphs-conditional-switch-nodes) execute the zero-indexed nth body graph once if the condition value is equal to n.  If the condition value does not correspond to a body graph, no body graph is launched.

A condition value is accessed by a [conditional handle](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#cuda-graphs-conditional-handles) , which must be created before the node. The condition value can be set by device code using `cudaGraphSetConditional()`. A default value, applied on each graph launch, can also be specified when the handle is created.

When the conditional node is created, an empty graph is created and the handle is returned to the user so that the graph can be populated.  This conditional body graph can be populated using either the [graph APIs](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#cuda-graphs-creating-a-graph-using-graph-apis) or [cudaStreamBeginCaptureToGraph()](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#cuda-graphs-creating-a-graph-using-stream-capture).

Conditional nodes can be nested.
