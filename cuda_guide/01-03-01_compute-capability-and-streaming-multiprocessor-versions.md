---
title: "1.3.1. Compute Capability and Streaming Multiprocessor Versions"
section: "1.3.1"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/cuda-platform.html#compute-capability-and-streaming-multiprocessor-versions"
---

## [1.3.1. Compute Capability and Streaming Multiprocessor Versions](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction#compute-capability-and-streaming-multiprocessor-versions)[](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/#compute-capability-and-streaming-multiprocessor-versions "Permalink to this headline")

Every NVIDIA GPU has a _Compute Capability_ (CC) number, which indicates what features are supported by that GPU and specifies some hardware parameters for that GPU. These specifications are documented in the [Section 5.1](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html#compute-capabilities) appendix.  A list of all NVIDIA GPUs and their compute capabilities is maintained on the [CUDA GPU Compute Capability page](https://developer.nvidia.com/cuda-gpus).

Compute capability is denoted as a major and minor version number in the format X.Y where X is the major version number and Y is the minor version number. For example, CC 12.0 has a major version of 12 and a minor version of 0. The compute capability directly corresponds to the version number of the SM. For example, the SMs within a GPU of CC 12.0 have SM version *sm_120*. This version is used to label binaries.

[Section 5.1.1](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html#compute-capabilities-querying) shows how to query and determine the compute capability of the GPU(s) in a system.
