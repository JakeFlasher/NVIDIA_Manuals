---
title: "4.12.2.2. Use-case: Thread Block Clusters"
section: "4.12.2.2"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cluster-launch-control.html#use-case-thread-block-clusters"
---

### [4.12.2.2. Use-case: Thread Block Clusters](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#use-case-thread-block-clusters)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#use-case-thread-block-clusters "Permalink to this headline")

In the case of a [thread block clusters](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/intro-to-cuda-cpp.html#thread-block-clusters),
the thread block cancellation steps are the same as in a non-cluster setting,
with minor adjustments.
As in the non-cluster case, submitting a cancellation request from multiple
threads **within a cluster** is not recommended,
as this will attempt to cancel multiple clusters.

- The cancellation is submitted by a single cluster thread.
- The shared memory result of each cluster’s thread block will receive
the same (encoded) value of the cancelled thread block index
(i.e., the result value is multicasted).
The result received by all thread blocks corresponds to the local
block index `{0, 0, 0}` within a cluster.
Therefore, thread blocks within the cluster need to add
the local block index.
- Synchronization is performed by each cluster’s thread block using
a local `__shared__` memory barrier. Barrier operations must be performed
with the `ptx::scope_cluster` scope.
- Cancelling in the cluster case requires all the thread blocks to exist.
A user can guarantee that all thread blocks are running by using
`cg::cluster_group::sync()` from [sync](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/device-callable-apis.html#cg-api-sync-function) API.

The kernel below demonstrates the cluster launch control approach using thread block clusters.

```c++
#include <cooperative_groups.h>
#include <cuda/ptx>

namespace cg = cooperative_groups;
namespace ptx = cuda::ptx;

__global__ __cluster_dims__(2, 1, 1)
void kernel_cluster_launch_control (float* data, int n)
{
    // Cluster launch control initialization:
    __shared__ uint4 result;
    __shared__ uint64_t bar;
    int phase = 0;

    if (cg::thread_block::thread_rank() == 0) {
        ptx::mbarrier_init(&bar, 1);
        ptx::fence_mbarrier_init(ptx::sem_release, ptx::scope_cluster); // CGA-level fence.
    }

    // Prologue:
    float alpha = compute_scalar(); // Device function not shown in this code snippet.

    // Work-stealing loop:
    int bx = blockIdx.x; // Assuming 1D x-axis thread blocks.

    while (true) {
        // Protect result from overwrite in the next iteration,
        // (also ensure all thread blocks have started at 1st iteration):
        cg::cluster_group::sync();

        // Cancellation request by a single cluster thread:
        if (cg::cluster_group::thread_rank() == 0) {
            // Acquire write of result in the async proxy:
            ptx::fence_proxy_async_generic_sync_restrict(ptx::sem_acquire, ptx::space_cluster, ptx::scope_cluster);

            cg::invoke_one(cg::coalesced_threads(), [&](){ptx::clusterlaunchcontrol_try_cancel_multicast(&result, &bar);});
        }

        // Cancellation completion tracked by each thread block:
        if (cg::thread_block::thread_rank() == 0)
            ptx::mbarrier_arrive_expect_tx(ptx::sem_relaxed, ptx::scope_cluster, ptx::space_shared, &bar, sizeof(uint4));

        // Computation:
        int i = bx * blockDim.x + threadIdx.x;
        if (i < n)
            data[i] *= alpha;

        // Cancellation request synchronization:
        while (!ptx::mbarrier_try_wait_parity(ptx::sem_acquire, ptx::scope_cluster, &bar, phase))
        {}
        phase ^= 1;

        // Cancellation request decoding:
        bool success = ptx::clusterlaunchcontrol_query_cancel_is_canceled(result);
        if (!success)
            break;

        bx = ptx::clusterlaunchcontrol_query_cancel_get_first_ctaid_x<int>(result);
        bx += cg::cluster_group::block_index().x; // Add local offset.

        // Release read of result to the async proxy:
        ptx::fence_proxy_async_generic_sync_restrict(ptx::sem_release, ptx::space_shared, ptx::scope_cluster);
    }
}

// Launch: kernel_cluster_launch_control<<<1024, (n + 1023) / 1024>>>(data, n);
```
