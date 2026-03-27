---
title: "5.6.4.2.2.1.4. Constant Memory"
section: "5.6.4.2.2.1.4"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/device-callable-apis.html#device-callable-apis--constant-memory"
---

###### [5.6.4.2.2.1.4. Constant Memory](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#constant-memory)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#constant-memory "Permalink to this headline")

Constants may not be modified from the device. They may only be modified from the host, but the behavior of modifying a constant from the host while there is a concurrent grid that access that constant at any point during its lifetime is undefined.
