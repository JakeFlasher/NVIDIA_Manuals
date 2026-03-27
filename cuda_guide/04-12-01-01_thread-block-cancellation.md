---
title: "4.12.1.1. Thread Block Cancellation"
section: "4.12.1.1"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cluster-launch-control.html#thread-block-cancellation"
---

### [4.12.1.1. Thread Block Cancellation](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#thread-block-cancellation)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#thread-block-cancellation "Permalink to this headline")

The preferred way to use Cluster Launch Control is from a single thread,
i.e., one request at a time.

The cancellation process involves five steps:

- **Setup Phase** (Steps 1-2): Declare and initialize cancellation result and synchronization variables.
- **Work-Stealing Loop** (Steps 3-5): Execute repeatedly to request, synchronize, and process cancellation results.

1. Declare variables for thread block cancellation:

```c++
__shared__ uint4 result; // Request result.
__shared__ uint64_t bar; // Synchronization barrier.
int phase = 0;           // Synchronization barrier phase.
```
2. Initialize shared memory barrier with a single arrival count:

```c++
if (cg::thread_block::thread_rank() == 0)
    ptx::mbarrier_init(&bar, 1);
__syncthreads();
```
3. Submit asynchronous cancellation request by a single thread and
set transaction count:

```c++
if (cg::thread_block::thread_rank() == 0) {
    cg::invoke_one(cg::coalesced_threads(), [&](){ptx::clusterlaunchcontrol_try_cancel(&result, &bar);});
    ptx::mbarrier_arrive_expect_tx(ptx::sem_relaxed, ptx::scope_cta, ptx::space_shared, &bar, sizeof(uint4));
}
```

> **Note**
>
> Since thread block cancellation is a uniform instruction,
> it is recommended to submit it inside
> [invoke_one](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cooperative-groups.html#cooperative-groups-invoke-one) thread selector.
> This allows the compiler to optimize out the peeling loop.
4. Synchronize (complete) asynchronous cancellation request:

```c++
while (!ptx::mbarrier_try_wait_parity(&bar, phase))
{}
phase ^= 1;
```
5. Retrieve cancellation status and cancelled thread block index:

```c++
bool success = ptx::clusterlaunchcontrol_query_cancel_is_canceled(result);
if (success) {
    // Don't need all three for 1D/2D thread blocks:
    int bx = ptx::clusterlaunchcontrol_query_cancel_get_first_ctaid_x(result);
    int by = ptx::clusterlaunchcontrol_query_cancel_get_first_ctaid_y(result);
    int bz = ptx::clusterlaunchcontrol_query_cancel_get_first_ctaid_z(result);
}
```
6. Ensure visibility of shared memory operations between async and generic
[proxies](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#proxies),
and protect against data races between iterations of the work-stealing loop.
