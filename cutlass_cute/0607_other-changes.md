---
title: "Other changes"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/0x_gemm_tutorial.html#other-changes"
---

### [Other changes](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute#other-changes)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/#other-changes "Permalink to this headline")

In this version, we have also updated the shared memory layouts for `gemm_tn` from K-major to

```cpp
  // Define the smem layouts (static)
  auto sA = make_layout(make_shape (      bM,          bK),
                        make_stride(Int<1>{}, bM+Int<1>{}));  // (m,k) -> smem_idx; padded m-major
  auto sB = make_layout(make_shape (      bN,          bK),
                        make_stride(Int<1>{}, bN+Int<1>{}));  // (n,k) -> smem_idx; padded n-major
```

which produces M-major and N-major layouts, but they are padded to avoid shared memory bank conflicts. This simply improves the access pattern to and from shared memory and no other changes in the kernel are required.
