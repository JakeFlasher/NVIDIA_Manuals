---
title: "4.4.7.1. Memcpy Async Alignment Requirements"
section: "4.4.7.1"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cooperative-groups.html#memcpy-async-alignment-requirements"
---

### [4.4.7.1. Memcpy Async Alignment Requirements](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#memcpy-async-alignment-requirements)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#memcpy-async-alignment-requirements "Permalink to this headline")

`memcpy_async` is only asynchronous if the source is global memory and the destination is shared memory and both are at least 4-byte aligned.
For achieving best performance: an alignment of 16 bytes for both shared memory and global memory is recommended.
