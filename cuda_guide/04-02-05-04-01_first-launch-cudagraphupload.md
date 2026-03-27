---
title: "4.2.5.4.1. First Launch / cudaGraphUpload"
section: "4.2.5.4.1"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html#first-launch-cudagraphupload"
---

#### [4.2.5.4.1. First Launch / cudaGraphUpload](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#first-launch-cudagraphupload)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#first-launch-cudagraphupload "Permalink to this headline")

Physical memory cannot be allocated or mapped during graph instantiation because the stream in which the graph will execute is unknown. Mapping is done instead during graph launch. Calling `cudaGraphUpload` can separate out the cost of allocation from the launch by performing all mappings for that graph immediately and associating the graph with the upload stream. If the graph is then launched into the same stream, it will launch without any additional remapping.

Using different streams for graph upload and graph launch behaves similarly to switching streams, likely resulting in remap operations. In addition, unrelated memory pool management is permitted to pull memory from an idle stream, which could negate the impact of the uploads.
