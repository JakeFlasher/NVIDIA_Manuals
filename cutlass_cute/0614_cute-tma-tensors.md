---
title: "CuTe TMA Tensors"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/0z_tma_tensors.html#cute-tma-tensors"
---

# [CuTe TMA Tensors](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute#cute-tma-tensors)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/#cute-tma-tensors "Permalink to this headline")

Along your travels, you may find strange looking CuTe Tensors that are printed as something like

```console
ArithTuple(0,_0,_0,_0) o ((_128,_64),2,3,1):((_1@0,_1@1),_64@1,_1@2,_1@3)
```

What is an `ArithTuple`? Are those tensor strides? What do those mean? What is this for?

This documentation intends to answer those questions and introduce some of the more advanced features of CuTe.
