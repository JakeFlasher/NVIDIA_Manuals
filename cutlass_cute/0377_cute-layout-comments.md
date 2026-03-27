---
title: "CuTe Layout Comments"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/programming_guidelines.html#cute-layout-comments"
---

### [CuTe Layout Comments](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#cute-layout-comments)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#cute-layout-comments "Permalink to this headline")

- Right-align tensor shape layout comments at column 120.
- If layout comment is too long do your best to align it.
- If layout comment is too long and there are many related tensors
that the reader should read together,
try to align the layout comments of related tensors.

Here are a couple examples.

```c++
Tensor mC = make_tensor(make_gmem_ptr(params.ptr_C), make_shape(M,N), params.dC);                              // (M,N)
Tensor mD = make_tensor(make_gmem_ptr(params.ptr_D), make_shape(M,N), params.dD);                              // (M,N)
Tensor mAux = make_tensor(make_gmem_ptr(params.ptr_Aux), make_shape(M,N), params.dAux);                        // (M,N)

auto thr_mma = tiled_mma.get_thread_slice(thread_idx);
Tensor tCgD = thr_mma.partition_C(gD);                                                             // (VEC,THR_M,THR_N)
Tensor tCgC = thr_mma.partition_C(gC);                                                             // (VEC,THR_M,THR_N)
Tensor tCgAux = thr_mma.partition_C(gAux);                                                         // (VEC,THR_M,THR_N)
```

```c++
Tensor my_tensor = make_tensor<Type>(Layout<Shape<_2,_2>{}, Stride<_1,_2>>{});                           // (2,2):(1,2)

// Related tensors
Tensor my_tensor1 = make_tensor<Type>(ThisIsAVeryComplicatedLayoutWithAVeryLongName);         // ((Mode0_0,Mode0_1,Mode0_2),Mode1,Mode2,Mode3)
Tensor my_tensor2_related = make_tensor<Type>(ThisIsAVeryComplicatedLayoutWithAVeryLongName); // ((Mode0_0,Mode0_1,Mode0_2),Mode1,Mode2,Mode3)
```
