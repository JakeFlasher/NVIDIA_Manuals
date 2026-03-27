---
title: "5.2.3.9. CUDA_AUTO_BOOST [[deprecated]]"
section: "5.2.3.9"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/environment-variables.html#cuda-auto-boost-deprecated"
---

### [5.2.3.9. CUDA_AUTO_BOOST [[deprecated]]](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#cuda-auto-boost-deprecated)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#cuda-auto-boost-deprecated "Permalink to this headline")

The environment variable affects the GPU clock “auto boost” behavior, namely dynamic clock boosting.
It overrides the “auto boost” option of the `nvidia-smi` tool, namely `nvidia-smi --auto-boost-default=0`.

> **Note**
>
> This environment variable is deprecated. It is strongly suggested to use `nvidia-smi --applications-clocks=<memory,graphics>` or the [NVML API](https://docs.nvidia.com/deploy/nvml-api/group__nvmlDeviceCommands.html#group__nvmlDeviceCommands) instead of the `CUDA_AUTO_BOOST` environment variable.
