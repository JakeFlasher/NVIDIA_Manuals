---
title: "5.6.4.2.2.1.1. Device and Constant Memory"
section: "5.6.4.2.2.1.1"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/device-callable-apis.html#device-and-constant-memory"
---

###### [5.6.4.2.2.1.1. Device and Constant Memory](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#device-and-constant-memory)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#device-and-constant-memory "Permalink to this headline")

Memory declared at file scope with `__device__` or `__constant__` memory space specifiers behaves identically when using the device runtime. All kernels may read or write device variables, whether the kernel was initially launched by the host or device runtime. Equivalently, all kernels will have the same view of `__constant__`s as declared at the module scope.
