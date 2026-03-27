---
title: "3.5.2.3. CUDA Graphs"
section: "3.5.2.3"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/feature-survey.html#cuda-graphs"
---

### [3.5.2.3. CUDA Graphs](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced#cuda-graphs)[](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/#cuda-graphs "Permalink to this headline")

[CUDA graphs](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html#cuda-graphs) enable an application to specify a sequence of CUDA operations such as kernel launches or memory copies and the dependencies between these operations so that they can be executed efficiently on the GPU. Similar behavior can be attained by using [CUDA streams](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html#cuda-streams), and indeed one of the mechanisms for creating a graph is called [stream capture](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html#cuda-graphs-creating-a-graph-using-stream-capture), which enables the operations on a stream to be recorded into a CUDA graph. Graphs can also be created using the [CUDA graphs API](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html#cuda-graphs-creating-a-graph-using-graph-apis).

Once a graph has been created, it can be instantiated and executed many times. This is useful for specifying workloads that will be repeated. Graphs offer some performance benefits in reducing CPU launch costs associated with invoking CUDA operations as well as enabling optimizations only available when the whole workload is specified in advance.

[Section 4.2](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html#cuda-graphs) describes and demonstrates how to use CUDA graphs.
