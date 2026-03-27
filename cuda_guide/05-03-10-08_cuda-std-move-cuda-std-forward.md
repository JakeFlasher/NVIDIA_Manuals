---
title: "5.3.10.8. [cuda::]std::move, [cuda::]std::forward"
section: "5.3.10.8"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-support.html#cuda-std-move-cuda-std-forward"
---

### [5.3.10.8. [cuda::]std::move, [cuda::]std::forward](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#cuda-std-move-cuda-std-forward)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#cuda-std-move-cuda-std-forward "Permalink to this headline")

By default, the CUDA compiler implicitly considers `std::move` and `std::forward` function templates to have `__host__ __device__` execution space specifiers, and therefore they can be invoked directly from device code. The `nvcc` flag `--no-host-device-move-forward` disables this behavior; `std::move` and `std::forward` will then be considered as `__host__` functions and will not be directly invocable from device code.

> **Hint**
>
> `cuda::std::move` and `cuda::std::forward` on the contrary always have `__host__ __device__` execution space.
