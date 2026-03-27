---
title: "Tensor Creation"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/03_tensor.html#tensor-creation"
---

## [Tensor Creation](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute#tensor-creation)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/#tensor-creation "Permalink to this headline")

`Tensor`s can be constructed as owning or nonowning.

“Owning” `Tensor`s behave like `std::array`.
When you copy the `Tensor`, you (deep-)copy its elements,
and the `Tensor`’s destructor deallocates the array of elements.

“Nonowning” `Tensor`’s behave like a (raw) pointer.
Copying the `Tensor` doesn’t copy the elements,
and destroying the `Tensor` doesn’t deallocate the array of elements.

This has implications for developers of generic `Tensor` algorithms.
For example, input `Tensor` parameters of a function
should be passed by reference or const reference,
because passing a `Tensor` by value
may or may not make a deep copy of the `Tensor`’s elements.
