---
title: "Dependent kernel launches"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/dependent_kernel_launch.html#dependent-kernel-launches"
---

# [Dependent kernel launches](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#dependent-kernel-launches)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#dependent-kernel-launches "Permalink to this headline")

The Hopper and Blackwell architectures supports a new feature through which two kernels in the same stream can
overlap their execution, named
[Programmatic Dependent Launch (PDL)](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programmatic-dependent-launch-and-synchronization).
This allows kernels with conflict in global memory to programmatically and safely overlap portions
of their execution. Primary kernel can signal it is about to finish execution, and the next kernel is expected to
programmatically wait on the previous kernel to finish flushing its memory.

We enable PDL by setting a flag through the extended CUDA launch APIs. All CUTLASS kernels with PDL support
will wait on the prior kernel to flush its output to memory and signal the next kernel to start. This means
they can safely be dropped in with any other set of kernels using PDL as long as they also adhere to waiting on
the prior to flush its memory as well.

For more information, we refer you to the [PDL section in the CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programmatic-dependent-launch-and-synchronization).
