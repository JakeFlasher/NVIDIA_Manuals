---
title: "ArithTupleIterators and ArithTuples"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/0z_tma_tensors.html#arithtupleiterators-and-arithtuples"
---

### [ArithTupleIterators and ArithTuples](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute#arithtupleiterators-and-arithtuples)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/#arithtupleiterators-and-arithtuples "Permalink to this headline")

First, we build a `counting_iterator` equivalent for TMA coordinates. It should support

- dereference to a TMA coordinate, and
- offset by another TMA coordinate.

We’ll call this an `ArithmeticTupleIterator`. It stores a coordinate (a tuple of integers) that is represented as an `ArithmeticTuple`. The `ArithmeticTuple` is simply a (public subclass of) `cute::tuple` that has an overloaded `operator+` so that it can be offset by another tuple. The sum of two tuples is the tuple of the sum of the elements.

Now similar to `counting_iterator<int>(42)` we can create an implicit “iterator” (but without increment or other common iterator operations) over tuples that can be dereferenced and offset by other tuples

```cpp
ArithmeticTupleIterator citer_1 = make_inttuple_iter(42, Int<2>{}, Int<7>{});
ArithmeticTupleIterator citer_2 = citer_1 + make_tuple(Int<0>{}, 5, Int<2>{});
print(*citer_2);
```

which outputs

```console
(42,7,_9)
```

A TMA Tensor can use an iterator like this to store the current TMA coordinate “offset”. The “offset” here is in quotes because it’s clearly not a normal 1-D array offset or pointer.

In summary, one creates a TMA descriptor for the _whole global memory tensor_. The TMA descriptor defines a view into that tensor and the instruction takes TMA coordinates into that view. In order to generate and track those TMA coordinates, we define an implicit CuTe Tensor of TMA coordinates that can be tiled, sliced, and partitioned the exact same way as an ordinary CuTe Tensor.

We can now track and offset TMA coordinates with this iterator, but how do we get CuTe Layouts to generate non-integer offsets?
