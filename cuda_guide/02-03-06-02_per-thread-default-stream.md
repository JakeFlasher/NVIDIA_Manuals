---
title: "2.3.6.2. Per-thread Default Stream"
section: "2.3.6.2"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html#per-thread-default-stream"
---

### [2.3.6.2. Per-thread Default Stream](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics#per-thread-default-stream)[](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/#per-thread-default-stream "Permalink to this headline")

Starting in CUDA-7, CUDA allows for each host thread to have its own independent default stream, rather than the shared legacy default stream. In order to enable this behavior one must either use the *nvcc* compiler option `--default-stream per-thread` or define the `CUDA_API_PER_THREAD_DEFAULT_STREAM` preprocessor macro. When this behavior is enabled, each host thread will have its own independent default stream which will not synchronize with other streams in the same way the legacy default stream does. In such a situation the [legacy default stream example](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/#legacy-default-stream-example) will now exhibit the same synchronization behavior as the [non-blocking stream example](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/#non-blocking-stream-example).
