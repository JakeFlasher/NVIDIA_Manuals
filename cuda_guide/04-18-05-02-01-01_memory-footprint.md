---
title: "4.18.5.2.1.1. Memory Footprint"
section: "4.18.5.2.1.1"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/dynamic-parallelism.html#memory-footprint"
---

##### [4.18.5.2.1.1. Memory Footprint](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#memory-footprint)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#memory-footprint "Permalink to this headline")

The device runtime system software reserves memory for various management purposes, in particular a reservation for tracking pending grid launches. Configuration controls are available to reduce the size of this reservation in exchange for certain launch limitations. See [Configuration Options](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/device-callable-apis.html#device-runtime-configuration-options), below, for details.
