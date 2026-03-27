---
title: "Linear combinations of strides"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/0z_tma_tensors.html#linear-combinations-of-strides"
---

#### [Linear combinations of strides](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute#linear-combinations-of-strides)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/#linear-combinations-of-strides "Permalink to this headline")

Layouts work by taking the inner product
of the natural coordinate with their strides.
For strides made of integer elements, e.g., `(1,100)`,
the inner product of the input coordinate `(i,j)`
and the stride is `i + 100j`.
Offsetting an “ordinary” tensor’s pointer and this index
gives the pointer to the tensor element at `(i,j)`.

For strides of basis elements, we still compute the inner product of the natural coordinate with the strides.
For example, if the stride is `(1@0,1@1)`,
then the inner product of the input coordinate `(i,j)`
with the strides is `i@0 + j@1 = (i,j)`.
That translates into the (TMA) coordinate `(i,j)`.
If we wanted to reverse the coordinates,
then we could use `(1@1,1@0)` as the stride.
Evaluating the layout would give `i@1 + j@0 = (j,i)`.

A linear combination of basis elements
can be interpreted as a possibly multidimensional and hierarchical coordinate.
For instance, `2*2@1@0 + 3*1@1 + 4*5@1 + 7*1@0@0`
means `((0,4,...),0,...) + (0,3,0,...) + (0,20,0,...) + ((7,...),...) = ((7,4,...),23,...)`
and can be interpreted as the coordinate `((7,4),23)`.

Thus, linear combinations of these strides can be used to generate TMA coordinates.
These coordinates, in turn, can be used to offset TMA coordinate iterators.
