---
title: "3.2. Advanced Kernel Programming"
section: "3.2"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html#advanced-kernel-programming"
---

# [3.2. Advanced Kernel Programming](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced#advanced-kernel-programming)[](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/#advanced-kernel-programming "Permalink to this headline")

This chapter will first take a deeper dive into the hardware model of NVIDIA GPUs, and then introduce some of the more advanced features available in CUDA kernel code aimed at improving kernel performance. This chapter will introduce some concepts related to thread scopes, asynchronous execution, and the associated synchronization primitives. These conceptual discussions provide a necessary foundation for some of the advanced performance features available within kernel code.

Detailed descriptions for some of these features are contained in chapters dedicated to the features in the next part of this programming guide.

- [Advanced synchronization primitives](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/#advanced-kernels-advanced-sync-primitives) introduced in this chapter, are covered completely in [Section 4.9](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-barriers.html#asynchronous-barriers) and [Section 4.10](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/pipelines.html#pipelines).
- [Asynchronous data copies](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/#advanced-kernels-async-copies), including the tensor memory accelerator (TMA), are introduced in this chapter and covered completely in [Section 4.11](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html#async-copies).
