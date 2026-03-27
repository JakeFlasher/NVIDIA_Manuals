---
title: "Computing Composition"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/02_layout_algebra.html#computing-composition"
---

### [Computing Composition](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute#computing-composition)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/#computing-composition "Permalink to this headline")

First, a few observations:

- `B = (B_0, B_1, ...)`. A layout can be expressed as the concatenation of its sublayouts.
- `A o B = A o (B_0, B_1, ...) = (A o B_0, A o B_1, ...)`. When `B` is injective, composition is left-distributive with concatenation.

With the above, we can assume without loss of generality that `B = s:d` is a layout with integral shape and stride. We can also assume that `A` is a flattened, coalesced layout.

When `A` is integral, `A = a:b`, the result is rather trivial: `R = A o B = a:b o s:d = s:(b*d)`. Here, the result of the composition `R` are the first `s` elements of `A` strided by `d`.

When `A` is multimodal, we need to be more careful. Put into words, `A o B = A o s:d`, for integral `s` and `d` means that we want to:

1. Determine a layout that produces every `d`th element of `A`.

The shape of this intermediate layout can be calculated by progressively “dividing out” the first `d` elements from the shape of `A` starting from the left.

For example,

- `(6,2) /  2 => (3,2)`
- `(6,2) /  3 => (2,2)`
- `(6,2) /  6 => (1,2)`
- `(6,2) / 12 => (1,1)`
- `(3,6,2,8) /  3 => (1,6,2,8)`
- `(3,6,2,8) /  6 => (1,3,2,8)`
- `(3,6,2,8) /  9 => (1,2,2,8)`
- `(3,6,2,8) / 72 => (1,1,1,4)`

To compute the strides of the strided layout, the residues of the above operation are used to scale the strides of `A`. For instance, the last example `(3,6,2,8):(w,x,y,z) / 72` with strides `(w,x,y,z)` produces `(72*w,24*x,4*x,2*z)` as the strides of the strided layout.

As you may have noticed, we can only divide shapes by certain values and get a sensible result. This is called the **stride divisibility condition** and is statically checked in CuTe when possible.

2. Keep the first `s` elements of the newly strided `A` so that the result has a compatible shape with `B`. This can be computed by “modding out” the first `s` elements from the shape of `A` starting from the left.

For example,

- `(6,2) %  2 => (2,1)`
- `(6,2) %  3 => (3,1)`
- `(6,2) %  6 => (6,1)`
- `(6,2) % 12 => (6,2)`
- `(3,6,2,8) %  6 => (3,2,1,1)`
- `(3,6,2,8) %  9 => (3,3,1,1)`
- `(1,2,2,8) %  2 => (1,2,1,1)`
- `(1,2,2,8) % 16 => (1,2,2,4)`

This operation causes the result to have a shape that is compatible with `B`.

Again, this operation must satisfy a **shape divisibility condition** to yield a sensible result and is statically checked in CuTe when possible.

From the above examples, we can construct the composition `(3,6,2,8):(w,x,y,z) o 16:9 = (1,2,2,4):(9*w,3*x,y,z)`.

---
