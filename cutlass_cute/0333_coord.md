---
title: "Coord"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/fundamental_types.html#coord"
---

### [Coord](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#coord)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#coord "Permalink to this headline")

```c++
template <
  int Rank,
  typename Index = int
>
class Coord;
```

`Coord<Rank, class T = int>` is a container used explicitly for defining logical coordinates in tensors of known rank. Traditional vector operators are defined such as `+`, `-`, and scalar multiplication `*` to simplify the creation of vector-valued expressions on tensor coordinates.

**Example:** Vector operations on coordinates.

```c++
Coord<2> compute_offset(Coord<2> const & base) {

  Coord<2> stride = make_Coord(1, kM);

  return base + stride * make_Coord(threadIdx.x, threadIdx.y);
}
```

Instances of `Coord<>` are used throughout CUTLASS to compute indices into tensors. Frequently, the dimensions of tensors of known layouts may be given names such as “rows” or “columns”. To clarify the code, we have implemented several classes derived from `Coord<>` with accessors for each coordinate member.

Such classes include:

```c++
struct MatrixCoord : public Coord<2> {
  Index & row();
  Index & column();
};
```

and

```c++
struct Tensor4DCoord : public Coord<4> {
  Index & n();
  Index & h();
  Index & w();
  Index & c();
};
```
