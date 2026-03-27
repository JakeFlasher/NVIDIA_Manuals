---
title: "4.12.2.1. Use-case: Thread Blocks"
section: "4.12.2.1"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cluster-launch-control.html#use-case-thread-blocks"
---

### [4.12.2.1. Use-case: Thread Blocks](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#use-case-thread-blocks)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#use-case-thread-blocks "Permalink to this headline")

The three kernels below demonstrate the _Fixed Work per Thread Block_,
_Fixed Number of Thread Blocks_, and _Cluster Launch Control_ approaches
for vector-scalar multiplication
\(\(\overline{v} := \alpha \overline{v}\)\).

- Fixed Work per Thread Block:

```c++
__global__
void kernel_fixed_work (float* data, int n)
{
    // Prologue:
    float alpha = compute_scalar();
    // Computation:
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        data[i] *= alpha;
}
// Launch: kernel_fixed_work<<<1024, (n + 1023) / 1024>>>(data, n);
```
- Fixed Number of Thread Blocks:

```c++
__global__
void kernel_fixed_blocks (float* data, int n)
{
    // Prologue:
    float alpha = compute_scalar();
    // Computation:
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < n) {
        data[i] *= alpha;
        i += gridDim.x * blockDim.x;
    }
}
// Launch: kernel_fixed_blocks<<<1024, SM_COUNT>>>(data, n);
```
- Cluster Launch Control:

```c++
#include <cooperative_groups.h>
#include <cuda/ptx>
namespace cg = cooperative_groups;
namespace ptx = cuda::ptx;
__global__
void kernel_cluster_launch_control (float* data, int n)
{
    // Cluster launch control initialization:
    __shared__ uint4 result;
    __shared__ uint64_t bar;
    int phase = 0;
    if (cg::thread_block::thread_rank() == 0)
        ptx::mbarrier_init(&bar, 1);
    // Prologue:
    float alpha = compute_scalar(); // Device function not shown in this code snippet.
    // Work-stealing loop:
    int bx = blockIdx.x; // Assuming 1D x-axis thread blocks.
    while (true) {
        // Protect result from overwrite in the next iteration,
        // (also ensure barrier initialization at 1st iteration):
        __syncthreads();
        // Cancellation request:
        if (cg::thread_block::thread_rank() == 0) {
            // Acquire write of result in the async proxy:
            ptx::fence_proxy_async_generic_sync_restrict(ptx::sem_acquire, ptx::space_cluster, ptx::scope_cluster);
            cg::invoke_one(cg::coalesced_threads(), [&](){ptx::clusterlaunchcontrol_try_cancel(&result, &bar);});
            ptx::mbarrier_arrive_expect_tx(ptx::sem_relaxed, ptx::scope_cta, ptx::space_shared, &bar, sizeof(uint4));
        }
        // Computation:
        int i = bx * blockDim.x + threadIdx.x;
        if (i < n)
            data[i] *= alpha;
        // Cancellation request synchronization:
        while (!ptx::mbarrier_try_wait_parity(ptx::sem_acquire, ptx::scope_cta, &bar, phase))
        {}
        phase ^= 1;
        // Cancellation request decoding:
        bool success = ptx::clusterlaunchcontrol_query_cancel_is_canceled(result);
        if (!success)
            break;
        bx = ptx::clusterlaunchcontrol_query_cancel_get_first_ctaid_x<int>(result);
        // Release read of result to the async proxy:
        ptx::fence_proxy_async_generic_sync_restrict(ptx::sem_release, ptx::space_shared, ptx::scope_cluster);
    }
}
// Launch: kernel_cluster_launch_control<<<1024, (n + 1023) / 1024>>>(data, n);
```
