---
title: "4.5.1. Background"
section: "4.5.1"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html#background"
---

## [4.5.1. Background](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#background)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#background "Permalink to this headline")

A CUDA application utilizes the GPU by launching and executing multiple kernels on it.
A typical GPU activity timeline is shown in [Figure 39](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#gpu-activity).

![GPU activity timeline](images/___________1.png)

Figure 39 GPU activity timeline[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#gpu-activity "Link to this image")

Here, `secondary_kernel` is launched after `primary_kernel` finishes its execution.
Serialized execution is usually necessary because `secondary_kernel` depends on result data
produced by `primary_kernel`. If `secondary_kernel` has no dependency on `primary_kernel`,
both of them can be launched concurrently by using [CUDA Streams](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html#cuda-streams).
Even if `secondary_kernel` is dependent on `primary_kernel`, there is some potential for
concurrent execution. For example, almost all the kernels have
some sort of _preamble_ section during which tasks such as zeroing buffers or loading
constant values are performed.

![Preamble section of ``secondary_kernel``](images/___________2.png)

Figure 40 Preamble section of `secondary_kernel`[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#secondary-kernel-preamble "Link to this image")

[Figure 40](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#secondary-kernel-preamble) demonstrates the portion of `secondary_kernel` that could
be executed concurrently without impacting the application.
Note that concurrent launch also allows us to hide the launch latency of `secondary_kernel` behind
the execution of `primary_kernel`.

![Concurrent execution of ``primary_kernel`` and ``secondary_kernel``](images/___________3.png)

Figure 41 Concurrent execution of `primary_kernel` and `secondary_kernel`[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#preamble-overlap "Link to this image")

The concurrent launch and execution of `secondary_kernel` shown in [Figure 41](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#preamble-overlap) is
achievable using _Programmatic Dependent Launch_.

_Programmatic Dependent Launch_ introduces changes to the CUDA kernel launch APIs as explained in following section.
These APIs require at least compute capability 9.0 to provide overlapping execution.
