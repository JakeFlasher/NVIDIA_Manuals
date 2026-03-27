---
title: "4.16.5.3. Virtual Aliasing Support"
section: "4.16.5.3"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/virtual-memory-management.html#virtual-aliasing-support"
---

### [4.16.5.3. Virtual Aliasing Support](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#virtual-aliasing-support)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#virtual-aliasing-support "Permalink to this headline")

The virtual memory management APIs provide a way to create multiple virtual
memory mappings or “proxies” to the same allocation using multiple calls to
`cuMemMap` with different virtual addresses. This is called virtual aliasing.
Unless otherwise noted in the PTX ISA, writes to one proxy of the allocation
are considered inconsistent and incoherent with any other proxy of the same
memory until the writing device operation (grid launch, memcpy, memset, and so
on) completes. Grids present on the GPU prior to a writing device operation
but reading after the writing device operation completes are also considered
to have inconsistent and incoherent proxies.

For example, the following snippet is considered undefined, assuming device
pointers A and B are virtual aliases of the same memory allocation:

```c++
__global__ void foo(char *A, char *B) {
  *A = 0x1;
  printf("%d\n", *B);    // Undefined behavior!  *B can take on either
// the previous value or some value in-between.
}
```

The following is defined behavior, assuming these two kernels are ordered
monotonically (by streams or events).

```c++
__global__ void foo1(char *A) {
  *A = 0x1;
}

__global__ void foo2(char *B) {
  printf("%d\n", *B);    // *B == *A == 0x1 assuming foo2 waits for foo1
// to complete before launching
}

cudaMemcpyAsync(B, input, size, stream1);    // Aliases are allowed at
// operation boundaries
foo1<<<1,1,0,stream1>>>(A);                  // allowing foo1 to access A.
cudaEventRecord(event, stream1);
cudaStreamWaitEvent(stream2, event);
foo2<<<1,1,0,stream2>>>(B);
cudaStreamWaitEvent(stream3, event);
cudaMemcpyAsync(output, B, size, stream3);  // Both launches of foo2 and
                                            // cudaMemcpy (which both
                                            // read) wait for foo1 (which writes)
                                            // to complete before proceeding
```

If accessing same allocation through different “proxies” is required in the
same kernel, a `fence.proxy.alias` can be used between the two accesses. The
above example can thus be made legal with inline PTX assembly:

```c++
__global__ void foo(char *A, char *B) {
  *A = 0x1;
  cuda::ptx::fence_proxy_alias();
  printf("%d\n", *B);    // *B == *A == 0x1
}
```
