---
title: "5.6.4.9. ECC Errors"
section: "5.6.4.9"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/device-callable-apis.html#ecc-errors"
---

### [5.6.4.9. ECC Errors](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#ecc-errors)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#ecc-errors "Permalink to this headline")

No notification of ECC errors is available to code within a CUDA kernel. ECC errors are reported at the host side once the entire launch tree has completed. Any ECC errors which arise during execution of a nested program will either generate an exception or continue execution (depending upon error and configuration).
