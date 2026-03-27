---
title: "Implicit CuTe Tensors"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/0z_tma_tensors.html#implicit-cute-tensors"
---

### [Implicit CuTe Tensors](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute#implicit-cute-tensors)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/#implicit-cute-tensors "Permalink to this headline")

All CuTe Tensors are compositions of Layouts and Iterators. An ordinary global memory tensor’s iterator is its global memory pointer. However, a CuTe Tensor’s iterator doesn’t have to be a pointer; it can be any random-access iterator.

One example of such an iterator is a _counting iterator_.
This represents a possibly infinite sequence of integers that starts at some value.
We call the members of this sequence _implicit integers_,
because the sequence is not explicitly stored in memory.
The iterator just stores its current value.

We can use a counting iterator to create a tensor of implicit integers,

```cpp
Tensor A = make_tensor(counting_iterator<int>(42), make_shape(4,5));
print_tensor(A);
```

which outputs

```console
counting_iter(42) o (4,5):(_1,4):
   42   46   50   54   58
   43   47   51   55   59
   44   48   52   56   60
   45   49   53   57   61
```

This tensor maps logical coordinates to on-the-fly computed integers. Because it’s still a CuTe Tensor, it can still be tiled and partitioned and sliced just like a normal tensor by accumulating integer offsets into the iterator.

But the TMA doesn’t consume pointers or integers, it consumes coordinates. Can we make a tensor of implicit TMA
coordinates for the TMA instruction to consume? If so, then we could presumably also tile and partition and slice that tensor of coordinates so that we would always have the right TMA coordinate to give to the instruction.
