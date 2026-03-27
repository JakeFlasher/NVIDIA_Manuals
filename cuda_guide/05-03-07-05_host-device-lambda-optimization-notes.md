---
title: "5.3.7.5. Host-Device Lambda Optimization Notes"
section: "5.3.7.5"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-support.html#host-device-lambda-optimization-notes"
---

### [5.3.7.5. Host-Device Lambda Optimization Notes](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#host-device-lambda-optimization-notes)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#host-device-lambda-optimization-notes "Permalink to this headline")

Unlike device-only lambdas, host-device lambdas can be called from host code. As previously mentioned, the CUDA compiler replaces an extended lambda expression defined in host code with an instance of a named placeholder type. The placeholder type for an extended host-device lambda invokes the original lambda’s `operator()` with an indirect function call. The traits will always return false if extended lambda mode is not active.

The presence of an indirect function call may cause the host compiler to optimize an extended host-device lambda less than lambdas that are implicitly or explicitly `__host__` only. In the latter case, the host compiler can easily inline the lambda body into the calling context. However, when it encounters an extended host-device lambda, the host compiler may not be able to easily inline the original lambda body.
