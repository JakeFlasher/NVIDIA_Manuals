---
title: "4.2.1.1. Node Types"
section: "4.2.1.1"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html#node-types"
---

### [4.2.1.1. Node Types](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#node-types)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#node-types "Permalink to this headline")

A graph node can be one of:

- kernel
- CPU function call
- memory copy
- memset
- empty node
- waiting on a [CUDA Event](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html#cuda-events)
- recording a [CUDA Event](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html#cuda-events)
- signalling an [external semaphore](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/graphics-interop.html#external-resource-interoperability)
- waiting on an [external semaphore](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/graphics-interop.html#external-resource-interoperability)
- [conditional node](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#cuda-graphs-conditional-graph-nodes)
- [memory node](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#cuda-graphs-graph-memory-nodes)
- child graph: To execute a separate nested graph, as shown in the following figure.

![Child Graph Example](images/____-______1.png)

Figure 21 Child Graph Example[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#cuda-graphs-node-types-fig-child-graph "Link to this image")
