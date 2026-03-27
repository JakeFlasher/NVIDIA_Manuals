---
title: "3.5.2.5. Lazy Loading"
section: "3.5.2.5"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/feature-survey.html#lazy-loading"
---

### [3.5.2.5. Lazy Loading](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced#lazy-loading)[](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/#lazy-loading "Permalink to this headline")

[Lazy loading](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/lazy-loading.html#lazy-loading) is a feature which allows control over how the JIT compiler operates at application startup. Applications which have many kernels which need to be JIT compiled from PTX to cubin may experience long startup times if all kernels are JIT compiled as part of application startup. The default behavior is that modules are not compiled until they are needed. This can be changed by the use of [environment variables](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/environment-variables.html#cuda-environment-variables), as detailed in [Section 4.7](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/lazy-loading.html#lazy-loading).
