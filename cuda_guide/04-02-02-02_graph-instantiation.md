---
title: "4.2.2.2. Graph Instantiation"
section: "4.2.2.2"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html#graph-instantiation"
---

### [4.2.2.2. Graph Instantiation](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#graph-instantiation)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#graph-instantiation "Permalink to this headline")

Once a graph has been created, either by the use of the graph API or stream capture, the graph must be instantiated to create an executable graph, which can then be launched.  Assuming the `cudaGraph_t graph` has been created successfully, the following code will instantiate the graph and create the executable graph `cudaGraphExec_t graphExec`:

```cuda
cudaGraphExec_t graphExec;
cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);
```
