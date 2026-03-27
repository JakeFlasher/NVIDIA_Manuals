---
title: "Can host-side work be overlapped with other device kernels in my application?"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/grouped_scheduler.html#can-host-side-work-be-overlapped-with-other-device-kernels-in-my-application"
---

### [Can host-side work be overlapped with other device kernels in my application?](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#can-host-side-work-be-overlapped-with-other-device-kernels-in-my-application)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#can-host-side-work-be-overlapped-with-other-device-kernels-in-my-application "Permalink to this headline")

For example, if a grouped GEMM is used as the Nth layer in a neural network,
host-side precomputation for the grouped GEMM can potentially be overlapped
with device-side work for layer N-1. In this case `kHostPrecompute` is likely
a good fit.
