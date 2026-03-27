---
title: "5.6.4.5. Device Management"
section: "5.6.4.5"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/device-callable-apis.html#device-callable-apis--device-management"
---

### [5.6.4.5. Device Management](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#device-management)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#device-management "Permalink to this headline")

There is no multi-GPU support from the device runtime; the device runtime is only capable of operating on the device upon which it is currently executing. It is permitted, however, to query properties for any CUDA capable device in the system.
