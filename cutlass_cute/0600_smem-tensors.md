---
title: "SMEM tensors"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/0x_gemm_tutorial.html#smem-tensors"
---

### [SMEM tensors](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute#smem-tensors)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/#smem-tensors "Permalink to this headline")

The shared memory layouts that are used to hold the tiles of data for A and B are also passed in as the parameters `ASmemLayout sA_layout` and `BSmemLayout sB_layout`.

These are defined in `gemm_nt` as

```c++
  // Define the smem layouts (static)
  auto sA = make_layout(make_shape(bM, bK));   // (m,k) -> smem_idx; m-major
  auto sB = make_layout(make_shape(bN, bK));   // (n,k) -> smem_idx; n-major
```

which produces simple M-major and N-major layouts. In `gemm_tn` these are

```cpp
  // Define the smem layouts (static)
  auto sA = make_layout(make_shape(bM,bK), LayoutRight{});   // (m,k) -> smem_idx; k-major
  auto sB = make_layout(make_shape(bN,bK), LayoutRight{});   // (n,k) -> smem_idx; k-major
```

which produces simple K-major layouts.

As is evident, these smem layouts can be almost anything. Inside the kernel, they are checked for only two properties: the shared memory layouts are static and they are the same top-level shape as the `CtaTiler`.

```cpp
  // Preconditions
  static_assert(is_static<ASmemLayout>::value);
  static_assert(is_static<BSmemLayout>::value);
  static_assert(is_static<CSmemLayout>::value);

  CUTE_STATIC_ASSERT_V(size<0>(ASmemLayout{}) == size<0>(cta_tiler));  // BLK_M
  CUTE_STATIC_ASSERT_V(size<0>(CSmemLayout{}) == size<0>(cta_tiler));  // BLK_M
  CUTE_STATIC_ASSERT_V(size<0>(BSmemLayout{}) == size<1>(cta_tiler));  // BLK_N
  CUTE_STATIC_ASSERT_V(size<1>(CSmemLayout{}) == size<1>(cta_tiler));  // BLK_N
  CUTE_STATIC_ASSERT_V(size<1>(ASmemLayout{}) == size<2>(cta_tiler));  // BLK_K
  CUTE_STATIC_ASSERT_V(size<1>(BSmemLayout{}) == size<2>(cta_tiler));  // BLK_K
```

Use of static layouts has a few advantages.

- Static layouts let us statically allocate shared memory as shown below.
- Static layouts are often more efficient and allow CuTe to dispatch to optimized implementations.
- Static layouts makes it easier to prove correctness of the algorithm and provide checks like the above – the smem layout sizes are the same as the CTA tile sizes.

As stated, the shared memory layouts can be anything that satisfy those conditions. Optimizing kernels like these is often performed by finding a good shared memory layout that provides good access patterns for both the writes to and the reads from shared memory. This includes the ability to vectorize reads and writes as well as avoid shared memory bank conflicts.

With the static smem layouts, the `gemm_device` kernel can allocate the required shared memory and create the smem `Tensor`s.

```cpp
  // Shared memory buffers
  __shared__ TA smemA[cosize_v<ASmemLayout>];
  __shared__ TB smemB[cosize_v<BSmemLayout>];
  Tensor sA = make_tensor(make_smem_ptr(smemA), sA_layout);  // (BLK_M,BLK_K)
  Tensor sB = make_tensor(make_smem_ptr(smemB), sB_layout);  // (BLK_N,BLK_K)
```

Note how the shared memory allocation depends only on the data type and the layout. What’s a `cosize`? Because a `Layout` is a function, we can speak of its domain and codomain. The `size` of a layout is the size of its domain and the `cosize` of a layout is the size of its codomain. If we want to allocate an array for which all the offsets produced by a layout are valid, then we can use the `cosize` of the layout as the length of the array (in units of elements).
