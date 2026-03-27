---
title: "Grouping and flattening"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/01_layout.html#grouping-and-flattening"
---

### [Grouping and flattening](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute#grouping-and-flattening)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/#grouping-and-flattening "Permalink to this headline")

Layout modes can be grouped with `group<ModeBegin, ModeEnd>` and flattened with `flatten`.

```cpp
Layout a = Layout<Shape<_2,_3,_5,_7>>{};  // (_2,_3,_5,_7):(_1,_2,_6,_30)
Layout b = group<0,2>(a);                 // ((_2,_3),_5,_7):((_1,_2),_6,_30)
Layout c = group<1,3>(b);                 // ((_2,_3),(_5,_7)):((_1,_2),(_6,_30))
Layout f = flatten(b);                    // (_2,_3,_5,_7):(_1,_2,_6,_30)
Layout e = flatten(c);                    // (_2,_3,_5,_7):(_1,_2,_6,_30)
```

Grouping, flattening, and reordering modes allows the reinterpretation of tensors in place as matrices, matrices as vectors, vectors as matrices, etc.
