---
title: "By-mode Coalesce"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/02_layout_algebra.html#by-mode-coalesce"
---

### [By-mode Coalesce](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute#by-mode-coalesce)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/#by-mode-coalesce "Permalink to this headline")

Obviously, sometimes we do care about the shape of our `Layout`, but would still like to coalesce. For example, I have a 2-D `Layout` and I would like the result to remain 2-D.

For this reason, there’s an overload of `coalesce` that takes an additional parameter

```cpp
// Apply coalesce at the terminals of trg_profile
Layout coalesce(Layout const& layout, IntTuple const& trg_profile)
```

which can be used as follows

```cpp
auto a = Layout<Shape <_2,Shape <_1,_6>>,
                Stride<_1,Stride<_6,_2>>>{};
auto result = coalesce(a, Step<_1,_1>{});   // (_2,_6):(_1,_2)
// Identical to
auto same_r = make_layout(coalesce(layout<0>(a)),
                          coalesce(layout<1>(a)));
```

This function is recursing into `Step<_1,_1>{}` and applying `coalesce` to the corresponding sublayout whenever it sees an integer (the values don’t matter, they’re just flags) rather than a tuple.

> This theme of defining an operation that treats a `Layout` as a “1-D” function from integers to integers and then generalizing to use it for an arbitrarily shaped layout will be a common one!
