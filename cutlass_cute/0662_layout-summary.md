---
title: "Summary:"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/layout.html#layout--summary"
---

## [Summary:](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#summary)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#summary "Permalink to this headline")

The design patterns described in this document form a hierarchy:

- `T *ptr;` is a pointer to a contiguous sequence of elements of type `T`
- `Layout layout;` is an object mapping an index space to a linear offset
- `TensorRef<T, Layout> ref(ptr, layout);` is an object pointing to an _unbounded_ tensor containing elements of type `T` and a layout of type `Layout`
- `TensorView<T, Layout> view(ref, extent);` is an object pointing to a _bounded_ tensor containing elements of type `T` and a layout of type `Layout`
