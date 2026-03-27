---
title: "4.2.2.3. Graph Execution"
section: "4.2.2.3"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html#graph-execution"
---

### [4.2.2.3. Graph Execution](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#graph-execution)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#graph-execution "Permalink to this headline")

After a graph has been created and instantiated to create an executable graph, it can be launched. Assuming the `cudaGraphExec_t graphExec` has been created successfully, the following code snippet will launch the graph into the specified stream:

```cuda
cudaGraphLaunch(graphExec, stream);
```

Pulling it all together and using the stream capture example from [Section 4.2.2.1.2](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#cuda-graphs-creating-a-graph-using-stream-capture), the following code snippet will create a graph, instantiate it, and launch it:

```cuda
cudaGraph_t graph;

cudaStreamBeginCapture(stream);

kernel_A<<< ..., stream >>>(...);
kernel_B<<< ..., stream >>>(...);
libraryCall(stream);
kernel_C<<< ..., stream >>>(...);

cudaStreamEndCapture(stream, &graph);

cudaGraphExec_t graphExec;
cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);
cudaGraphLaunch(graphExec, stream);
```
