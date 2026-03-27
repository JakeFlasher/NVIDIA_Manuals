---
title: "5.8.3.1. Dependencies"
section: "5.8.3.1"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cuda-cpp-execution-model.html#dependencies"
---

### [5.8.3.1. Dependencies](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#dependencies)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#dependencies "Permalink to this headline")

A device thread shall not start until all its dependencies have completed.

> [Note: Dependencies that prevent device threads from starting to make progress can be created, for example, via [CUDA Stream Commands](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html#cuda-streams).
> These may include dependencies on the completion of, among others, [CUDA Events](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html#cuda-events) and [CUDA Kernels](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/intro-to-cuda-cpp.html#kernels). - end note.]

<details>
<summary>Examples of CUDA API forward progress guarantees due to dependencies</summary>

```cuda
 1// Example: Execution.Model.Stream.0
 2// Allowed outcome: eventually, no thread makes progress.
 3// Rationale: while CUDA guarantees that one device thread makes progress, since there
 4// is no dependency between `first` and `second`, it does not guarantee which thread,
 5// and therefore it could always pick the device thread from `second`, which then never
 6// unblocks from the spin-loop.
 7// That is, `second` may starve `first`.
 8cuda::atomic<int, cuda::thread_scope_system> flag = 0;
 9__global__ void first() { flag.store(1, cuda::memory_order_relaxed); }
10__global__ void second() { while(flag.load(cuda::memory_order_relaxed) == 0) {} }
11int main() {
12    cudaHostRegister(&flag, sizeof(flag));
13    cudaStream_t s0, s1;
14    cudaStreamCreate(&s0);
15    cudaStreamCreate(&s1);
16    first<<<1,1,0,s0>>>();
17    second<<<1,1,0,s1>>>();
18    return cudaDeviceSynchronize();
19}
```

```cuda
 1// Example: Execution.Model.Stream.1
 2// Outcome: terminates.
 3// Rationale: same as Execution.Model.Stream.0, but this example has a stream dependency
 4// between first and second, which requires CUDA to run the grids in order.
 5cuda::atomic<int, cuda::thread_scope_system> flag = 0;
 6__global__ void first() { flag.store(1, cuda::memory_order_relaxed); }
 7__global__ void second() { while(flag.load(cuda::memory_order_relaxed) == 0) {} }
 8int main() {
 9    cudaHostRegister(&flag, sizeof(flag));
10    cudaStream_t s0;
11    cudaStreamCreate(&s0);
12    first<<<1,1,0,s0>>>();
13    second<<<1,1,0,s0>>>();
14    return cudaDeviceSynchronize();
15}
```

</details>
