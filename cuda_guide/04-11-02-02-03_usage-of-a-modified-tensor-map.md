---
title: "4.11.2.2.3. Usage of a Modified Tensor Map"
section: "4.11.2.2.3"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html#usage-of-a-modified-tensor-map"
---

#### [4.11.2.2.3. Usage of a Modified Tensor Map](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#usage-of-a-modified-tensor-map)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#usage-of-a-modified-tensor-map "Permalink to this headline")

In contrast to using a tensor map that is passed as a `const __grid_constant__` kernel parameter, using a tensor map in
global memory requires explicitly establishing a release-acquire pattern in the tensor map proxy between the threads
that modify the tensor map and the threads that use it.

The release part of the pattern was shown in the previous section. It is accomplished using
the [cuda::ptx::tensormap.cp_fenceproxy](https://nvidia.github.io/cccl/libcudacxx/ptx/instructions/tensormap_cp_fenceproxy.html) function.

The acquire part is accomplished using the [cuda::ptx::fence_proxy_tensormap_generic](https://nvidia.github.io/cccl/libcudacxx/ptx/instructions/fence.html)
function that wraps the [fence.proxy.tensormap::generic.acquire](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-membar-fence)
instruction. If the two threads participating in the release-acquire pattern are on the same device, the `.gpu` scope suffices. If the threads are on
different devices, the `.sys` scope must be used. Once a tensor map has been acquired by one thread, it can be used by other threads in the block
after sufficient synchronization, for example, using `__syncthreads()`. The thread that uses the tensor map and the thread that performs the fence
must be in the same block. That is, if the threads are in, for example, two different thread blocks of the same cluster, the same grid, or a
different kernel, synchronization APIs such as `cooperative_groups::cluster` or `grid_group::sync()` or stream-order synchronization do not
suffice to establish ordering for tensor map updates, that is, threads in these other thread blocks still need to acquire the tensor map proxy
at the right scope before using the updated tensor map. If there are no intermediate modifications, the fence does not have to be repeated
before each `cp.async.bulk.tensor` instruction.

The `fence` and subsequent use of the tensor map is shown in the following example.

```cuda
// Consumer of tensor map in global memory:
__global__ void consume_tensor_map(CUtensorMap* tensor_map) {
  // Fence acquire tensor map:
  ptx::n32_t<128> size_bytes;
  ptx::fence_proxy_tensormap_generic(ptx::sem_acquire, ptx::scope_sys, tensor_map, size_bytes);
  // Safe to use tensor_map after fence.

  __shared__ uint64_t bar;
  __shared__ alignas(128) char smem_buf[4][128];

  if (threadIdx.x == 0) {
    // Initialize barrier
    ptx::mbarrier_init(&bar, 1);
    // Issue TMA request
    ptx::cp_async_bulk_tensor(ptx::space_cluster, ptx::space_global, smem_buf, tensor_map, {0, 0}, &bar);
    // Arrive on barrier. Expect 4 * 128 bytes.
    ptx::mbarrier_arrive_expect_tx(ptx::sem_release, ptx::scope_cta, ptx::space_shared, &bar, sizeof(smem_buf));
  }
  const int parity = 0;
  // Wait for load to have completed
  while (!ptx::mbarrier_try_wait_parity(&bar, parity)) {}

  // print items:
  printf("Got:\n\n");
  for (int j = 0; j < 4; ++j) {
    for (int i = 0; i < 128; ++i) {
      printf("%3d ", smem_buf[j][i]);
      if (i % 32 == 31) { printf("\n"); };
    }
    printf("\n");
  }
}
```
