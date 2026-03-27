---
title: "4.5.3. Use in CUDA Graphs"
section: "4.5.3"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html#use-in-cuda-graphs"
---

## [4.5.3. Use in CUDA Graphs](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#use-in-cuda-graphs)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#use-in-cuda-graphs "Permalink to this headline")

Programmatic Dependent Launch can be used in [CUDA Graphs](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html#cuda-graphs) via [stream capture](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html#cuda-graphs-creating-a-graph-using-stream-capture) or
directly via [edge data](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html#cuda-graphs-edge-data). To program this feature in a CUDA Graph with edge data, use a `cudaGraphDependencyType`
value of `cudaGraphDependencyTypeProgrammatic` on an edge connecting two kernel nodes. This edge type makes the upstream kernel
visible to a `cudaGridDependencySynchronize()` in the downstream kernel. This type must be used with an outgoing port of
either `cudaGraphKernelNodePortLaunchCompletion` or `cudaGraphKernelNodePortProgrammatic`.

The resulting graph equivalents for stream capture are as follows:

| Stream code (abbreviated) | Resulting graph edge |
| --- | --- |
| ```c++ cudaLaunchAttribute attribute; attribute.id = cudaLaunchAttributeProgrammaticStreamSerialization; attribute.val.programmaticStreamSerializationAllowed = 1; ``` | ```c++ cudaGraphEdgeData edgeData; edgeData.type = cudaGraphDependencyTypeProgrammatic; edgeData.from_port = cudaGraphKernelNodePortProgrammatic; ``` |
| ```c++ cudaLaunchAttribute attribute; attribute.id = cudaLaunchAttributeProgrammaticEvent; attribute.val.programmaticEvent.triggerAtBlockStart = 0; ``` | ```c++ cudaGraphEdgeData edgeData; edgeData.type = cudaGraphDependencyTypeProgrammatic; edgeData.from_port = cudaGraphKernelNodePortProgrammatic; ``` |
| ```c++ cudaLaunchAttribute attribute; attribute.id = cudaLaunchAttributeProgrammaticEvent; attribute.val.programmaticEvent.triggerAtBlockStart = 1; ``` | ```c++ cudaGraphEdgeData edgeData; edgeData.type = cudaGraphDependencyTypeProgrammatic; edgeData.from_port = cudaGraphKernelNodePortLaunchCompletion; ``` |
