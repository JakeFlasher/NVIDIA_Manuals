---
title: "Layouts and Tensors"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/layout.html#layouts-and-tensors"
---

# [Layouts and Tensors](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#layouts-and-tensors)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#layouts-and-tensors "Permalink to this headline")

_Tensors_ are mathematical objects represented by a multidimensional array of numeric elements in memory.
These may define two dimensional matrices upon which classical linear algebra computations may be defined or
higher dimensional objects frequently used to structure data used by Deep Learning applications and frameworks.

This document describes design patterns used in CUTLASS to map logical index spaces onto memory (Layouts) and to
indirectly reference tensors in memory (TensorRef and TensorView objects).

As described, CUTLASS adheres to the following terminology which is consistent with the C++ Standard Library.

- _size_ (scalar): number of elements in a tensor
- _capacity_ (scalar): number of elements needed to represent tensor in memory (may be larger than _size_)
- _rank_ (scalar): number of logical dimensions describing tensor
- _extent_ (vector): size of each logical dimension in a tensor
