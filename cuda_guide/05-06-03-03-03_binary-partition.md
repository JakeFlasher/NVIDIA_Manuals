---
title: "5.6.3.3.3. binary_partition"
section: "5.6.3.3.3"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/device-callable-apis.html#binary-partition"
---

#### [5.6.3.3.3. binary_partition](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#binary-partition)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#binary-partition "Permalink to this headline")

```c++
coalesced_group binary_partition(const coalesced_group& g, bool pred);
```

```c++
template <unsigned int Size>
coalesced_group binary_partition(const thread_block_tile<Size>& g, bool pred);
```

The `binary_partition()` method is a collective operation that partitions the parent group into one-dimensional subgroups within which the threads are coalesced. The implementation will evaluate a predicate and assign threads that have the same value into the same group. This is a specialized form of `labeled_partition()`, where the label can only be 0 or 1.

The implementation may cause the calling thread to wait until all the members of the parent group have invoked the operation before resuming execution.

**Example:**

```c++
/// This example divides a 32-sized tile into a group with odd
/// numbers and a group with even numbers
_global__ void oddEven(int *inputArr) {
    auto block = cg::this_thread_block();
    auto tile32 = cg::tiled_partition<32>(block);

    // inputArr contains random integers
    int elem = inputArr[block.thread_rank()];
    // after this, tile32 is split into 2 groups,
    // a subtile where elem&1 is true and one where its false
    auto subtile = cg::binary_partition(tile32, (elem & 1));
}
```
