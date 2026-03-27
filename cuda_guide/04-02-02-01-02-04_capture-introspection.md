---
title: "4.2.2.1.2.4. Capture Introspection"
section: "4.2.2.1.2.4"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html#capture-introspection"
---

##### [4.2.2.1.2.4. Capture Introspection](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#capture-introspection)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#capture-introspection "Permalink to this headline")

Active stream capture operations can be inspected using `cudaStreamGetCaptureInfo()`.  This allows the user to obtain the status of the capture, a unique(per-process) ID for the capture, the underlying graph object, and dependencies/edge data for the next node to be captured in the stream.  This dependency information can be used to obtain a handle to the node(s) which were last captured in the stream.
