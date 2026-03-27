---
title: "5.8.3. CUDA APIs"
section: "5.8.3"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cuda-cpp-execution-model.html#cuda-apis"
---

## [5.8.3. CUDA APIs](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#cuda-apis)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#cuda-apis "Permalink to this headline")

A CUDA API call shall eventually either return or ensure at least one device thread makes progress.

CUDA query functions (e.g. [cudaStreamQuery](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g2021adeb17905c7ec2a3c1bf125c5435),
[cudaEventQuery](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html#group__CUDART__EVENT_1g2bf738909b4a059023537eaa29d8a5b7), etc.) shall not consistently
return `cudaErrorNotReady` without a device thread making progress.

> [Note: The device thread need not be “related” to the API call, e.g., an API operating on one stream or process may ensure progress of a device thread on another stream or process. - end note.]
>
>
> [Note: A simple but not sufficient method to test a program for CUDA API Forward Progress conformance is to run them with following environment variables set: `CUDA_DEVICE_MAX_CONNECTIONS=1 CUDA_LAUNCH_BLOCKING=1`, and then check that the program still terminates.
> If it does not, the program has a bug.
> This method is not sufficient because it does not catch all Forward Progress bugs, but it does catch many such bugs. - end note.]

<details>
<summary>Examples of CUDA API forward progress guarantees.</summary>

```cuda
 1// Example: Execution.Model.API.1
 2// Outcome: if no other device threads (e.g., from other processes) are making progress,
 3// this program terminates and returns cudaSuccess.
 4// Rationale: CUDA guarantees that if the device is empty:
 5// - `cudaDeviceSynchronize` eventually ensures that at least one device-thread makes progress, which implies that eventually `hello_world` grid and one of its device-threads start.
 6// - All thread-block threads eventually start (due to "if a device thread makes progress, all other threads in its thread-block cluster eventually make progress").
 7// - Once all threads in thread-block arrive at `__syncthreads` barrier, all waiting threads are unblocked.
 8// - Therefore all device threads eventually exit the `hello_world`` grid.
 9// - And `cudaDeviceSynchronize`` eventually unblocks.
10__global__ void hello_world() { __syncthreads(); }
11int main() {
12    hello_world<<<1,2>>>();
13    return (int)cudaDeviceSynchronize();
14}
```

```cuda
 1// Example: Execution.Model.API.2
 2// Allowed outcome: eventually, no thread makes progress.
 3// Rationale: the `cudaDeviceSynchronize` API below is only called if a device thread eventually makes progress and sets the flag.
 4// However, CUDA only guarantees that `producer` device thread eventually starts if the synchronization API is called.
 5// Therefore, the host thread may never be unblocked from the flag spin-loop.
 6cuda::atomic<int, cuda::thread_scope_system> flag = 0;
 7__global__ void producer() { flag.store(1); }
 8int main() {
 9    cudaHostRegister(&flag, sizeof(flag));
10    producer<<<1,1>>>();
11    while (flag.load() == 0);
12    return cudaDeviceSynchronize();
13}
```

```cuda
 1// Example: Execution.Model.API.3
 2// Allowed outcome: eventually, no thread makes progress.
 3// Rationale: same as Example.Model.API.2, with the addition that a single CUDA query API call does not guarantee
 4// the device thread eventually starts, only repeated CUDA query API calls do (see Execution.Model.API.4).
 5cuda::atomic<int, cuda::thread_scope_system> flag = 0;
 6__global__ void producer() { flag.store(1); }
 7int main() {
 8    cudaHostRegister(&flag, sizeof(flag));
 9    producer<<<1,1>>>();
10    (void)cudaStreamQuery(0);
11    while (flag.load() == 0);
12    return cudaDeviceSynchronize();
13}
```

```cuda
 1// Example: Execution.Model.API.4
 2// Outcome: terminates.
 3// Rationale: same as Execution.Model.API.3, but this example repeatedly calls
 4// a CUDA query API in within the flag spin-loop, which guarantees that the device thread
 5// eventually makes progress.
 6cuda::atomic<int, cuda::thread_scope_system> flag = 0;
 7__global__ void producer() { flag.store(1); }
 8int main() {
 9    cudaHostRegister(&flag, sizeof(flag));
10    producer<<<1,1>>>();
11    while (flag.load() == 0) {
12        (void)cudaStreamQuery(0);
13    }
14    return cudaDeviceSynchronize();
15}
```

</details>
