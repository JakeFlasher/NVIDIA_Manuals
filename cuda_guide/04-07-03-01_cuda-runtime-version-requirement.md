---
title: "4.7.3.1. CUDA Runtime Version Requirement"
section: "4.7.3.1"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/lazy-loading.html#cuda-runtime-version-requirement"
---

### [4.7.3.1. CUDA Runtime Version Requirement](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#cuda-runtime-version-requirement)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#cuda-runtime-version-requirement "Permalink to this headline")

Lazy loading is available starting in CUDA runtime version 11.7. As CUDA runtime is usually linked statically into programs and libraries, only programs and libraries from or compiled with CUDA 11.7+ toolkit will benefit from lazy loading. Libraries compiled using older CUDA runtime versions will load all modules eagerly.
