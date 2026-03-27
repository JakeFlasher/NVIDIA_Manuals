---
title: "4.10.5. Early Exit"
section: "4.10.5"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/pipelines.html#pipelines--early-exit"
---

## [4.10.5. Early Exit](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#early-exit)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#early-exit "Permalink to this headline")

When a thread that is participating in a pipeline must exit early, that thread must explicitly drop out of participation before exiting using `cuda::pipeline::quit()`. The remaining participating threads can proceed normally with subsequent operations.
