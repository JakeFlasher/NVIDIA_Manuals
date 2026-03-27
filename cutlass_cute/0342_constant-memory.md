---
title: "Constant Memory"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/programming_guidelines.html#constant-memory"
---

### [Constant Memory](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#constant-memory)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#constant-memory "Permalink to this headline")

Several CUTLASS template classes exhibit a pattern in which problem-specific internal state is known at kernel
launch time and remains invariant throughout the execution of a kernel. For example, tile iterators compute several
offsets based on the strides of the input tensor that is added to an internal pointer when loading the elements
of a tile. These are computed from the tensor stride and never updated; the per-thread internal state consists
only of the internal global memory pointer.

CUTLASS can take advantage of this CUDA grid-invariant property by constructing the object in host code and passing
a composed parameters structure to the kernel. This confers two benefits: (1.) invariant state is held in constant
memory, and (2.) there is no overhead to compute the initial state by each thread.

The design pattern in CUTLASS is for classes with nontrivial constructors to define `struct Params` as an inner class
which contains grid-invariant state. These should define a constructor and an `initialize()` method. The `Params`
structure should also include a data member corresponding to each data member in the parent class, so these too can
be properly constructed in host code. The parent class should define a constructor which accepts `Params const &` as
its first argument.
