---
title: "Thread-Value partitioning"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/03_tensor.html#thread-value-partitioning"
---

### [Thread-Value partitioning](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute#thread-value-partitioning)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/#thread-value-partitioning "Permalink to this headline")

Another common partitioning strategy is called a thread-value partitioning. In this pattern, we construct a `Layout` that represents the mapping of all threads (or any parallel agent) and all values that each thread will receive to coordinates of the target data. With `composition` the target data layout is transformed according to our TV-layout and then we can simply slice into the thread-mode of the result with our thread index.

```cpp
// Construct a TV-layout that maps 8 thread indices and 4 value indices
//   to 1D coordinates within a 4x8 tensor
// (T8,V4) -> (M4,N8)
auto tv_layout = Layout<Shape <Shape <_2,_4>,Shape <_2, _2>>,
                        Stride<Stride<_8,_1>,Stride<_4,_16>>>{}; // (8,4)

// Construct a 4x8 tensor with any layout
Tensor A = make_tensor<float>(Shape<_4,_8>{}, LayoutRight{});    // (4,8)
// Compose A with the tv_layout to transform its shape and order
Tensor tv = composition(A, tv_layout);                           // (8,4)
// Slice so each thread has 4 values in the shape and order that the tv_layout prescribes
Tensor  v = tv(threadIdx.x, _);                                  // (4)
```

![tv_layout.png](images/______-_____-_____________1.png)

The above image is a visual representation of the above code. An arbitrary 4x8 layout of data is composed with a specific 8x4 TV-layout that represents a partitioning pattern. The result of the composition is on the right where each threads’ values are arranged across each row. The bottom layout depicts the inverse TV layout which shows the mapping of 4x8 logical coordinates to the thread id and value id they will be mapped to.

To see how these partitioning patterns are constructed and used, see the [tutorial on building MMA Traits](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/0t_mma_atom.html).
