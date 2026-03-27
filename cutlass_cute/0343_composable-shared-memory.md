---
title: "Composable Shared Memory"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/programming_guidelines.html#composable-shared-memory"
---

### [Composable Shared Memory](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#composable-shared-memory)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#composable-shared-memory "Permalink to this headline")

Shared memory requires explicit effort by the programmer to allocate and de-allocate. CUTLASS follows the paradigm
introduced by [CUB](https://nvlabs.github.io/cub/) to define composed structures for storing data intended to be held
in shared memory. Any object requiring shared memory storage for itself or its data members should define a child
structure called `SharedStorage`. This holds data needed by the class and also instantiates `SharedStorage`
objects for each data member.

To be consistent, this pattern defines a convention in which classes define internal shared memory storage requirements.
Classes should consider all SharedStorage structures to be opaque other than their own child class. When the lifetimes
of child objects are known to be non-overlapping, `union`s may be used to alias multiple SharedStorage objects to the same
shared memory region and reduce overall shared memory capacity.  Developers should carefully note that C++ `union` rules
require that they only access the most recently written (“active”) member of the `union`; this differs from C rules.

For host to device ABI compatibility, inheritance from a class is only permitted if the superclass is unique to the
child class. This is most easily achieved by templating the parent class by the child class (CRTP).
