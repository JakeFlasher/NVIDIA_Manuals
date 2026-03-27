---
title: "3.5.2.4. Programmatic Dependent Launch"
section: "3.5.2.4"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/feature-survey.html#programmatic-dependent-launch"
---

### [3.5.2.4. Programmatic Dependent Launch](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced#programmatic-dependent-launch)[](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/#programmatic-dependent-launch "Permalink to this headline")

[Programmatic dependent launch](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html#programmatic-dependent-launch-and-synchronization) is a CUDA feature which allows a dependent kernel, i.e. a kernel which depends on the output of a prior kernel, to begin execution before the primary kernel on which it depends has completed. The dependent kernel can execute setup code and unrelated work up until it requires data from the primary kernel and block there. The primary kernel can signal when the data required by the dependent kernel is ready, which will release the dependent kernel to continue executing. This enables some overlap between the kernels which can help keep GPU utilization high while minimizing the latency of the critical data path. [Section 4.5](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html#programmatic-dependent-launch-and-synchronization) covers programmatic dependent launch.
