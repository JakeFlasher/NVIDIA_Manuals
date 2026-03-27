---
title: "3.1.6. Environment Variables"
section: "3.1.6"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-host-programming.html#environment-variables"
---

## [3.1.6. Environment Variables](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced#environment-variables)[](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/#environment-variables "Permalink to this headline")

CUDA provides various environment variables (see [Section 5.2](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/environment-variables.html#cuda-environment-variables)), which can affect execution and performance.
If they are not explicitly set, CUDA uses reasonable default values for them, but
special handling may be required on a per-case basis, e.g., for debugging purposes or to get improved performance.

For example, increasing the value of the `CUDA_DEVICE_MAX_CONNECTIONS` environment variable may be necessary to reduce the possibility that
independent work from different CUDA streams gets serialized due to false dependencies. Such false dependencies may be introduced when
the same underlying resource(s) are used.
It is recommended to start by using the default value and only explore the impact of this environment variable in case of performance issues (e.g., unexpected serialization of independent
work across CUDA streams that cannot be attributed to other factors like lack of available SM resources).
Worth noting that this environment variable has a different (lower) default value in case of MPS.

Similarly, setting the `CUDA_MODULE_LOADING` environment variable to `EAGER` may be preferable
for latency-sensitive applications, in order to move all overhead
due to module loading to the application initialization phase and outside its critical phase. The current default mode is lazy module loading.
In this default mode, a similar effect to eager module loading could be achieved
by adding “warm-up” calls of the various kernels during the application’s initialization phase, to force module loading to happen sooner.

Please refer to [CUDA Environment Variables](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/environment-variables.html#cuda-environment-variables) for more details about the various CUDA environment variables.
It is recommended that you set the environment variables to new values _before_ you launch the application; attempting to set them within your application
may have no effect.
