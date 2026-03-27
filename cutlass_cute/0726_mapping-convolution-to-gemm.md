---
title: "Mapping Convolution to GEMM"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/implicit_gemm_convolution.html#mapping-convolution-to-gemm"
---

## [Mapping Convolution to GEMM](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#mapping-convolution-to-gemm)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#mapping-convolution-to-gemm "Permalink to this headline")

The forward convolutional layer computes an output tensor _y = conv(x, w)_ where x(NHWC), w(KRSC), and y(NPQK)
are 4-D tensors.

This computation may be described by the following analytic function.

```console
y[n, p, q, k] = sum_c(sum_r(sum_s( x[n, f(p, r), g(q, s), c] * w[k, r, s, c] )))
```

where functions _f_ and _g_ are defined as follows.

```console
f(p, r) = p * stride_h + R - r - 1 + pad_h
g(q, s) = q * stride_w + S - s - 1 + pad_w
```

A [host](https://github.com/NVIDIA/cutlass/tree/main/tools/util/include/cutlass/util/reference/host/convolution.h) and [device](https://github.com/NVIDIA/cutlass/tree/main/tools/util/include/cutlass/util/reference/device/convolution.h)
reference implementation are provided in the CUTLASS Utilities.

This computation may be mapped to the elements of a matrix product as follows.

```console
C = gemm(A, B)
```

where

- A is a row-major matrix of extent _NHW_-by-_RSC_ containing activations
- B is a column-major matrix of extent _RSC_-by-_K_ containing filters
- C is a row-major matrix of extent _NPQ_-by-_K_ containing the output

Each element of the output matrix _Cij_ corresponds to an element in the output tensor y[n, p, q, k] according to
the following relation.

```console
y[n, p, q, k] = Cij
```

where

```console
i = q + Q * (p + P * n)
j = k
```

These relations may be inverted as follows.

```console
k = j

n = i / (PQ)
residual = i % (PQ)

p = residual / Q
q = residual % Q
```

The triple loop nest iterating over CRS to accumulate the result may also be linearized and mapped to the inner
GEMM _K_ dimension (not to be confused with the filter tensor dimension _K_) by the following relations.

```console
gemm_k = s + S * (r + R * c)
```

and inverse

```console
c = gemm_k / (RS)
residual = gemm_k % (RS)

r = residual / S
s = residual % S
```

Given these equations, a GEMM triple loop nest could be augmented with tensor indexing as follows.

```c++
int GEMM_M = N * P * Q;
int GEMM_N = K;
int GEMM_K = C * R * S;

for (int gemm_i = 0; gemm_i < GEMM_M; ++gemm_i) {
  for (int gemm_j = 0; gemm_j < GEMM_N; ++gemm_j) {

    int n = gemm_i / (PQ);
    int npq_residual = gemm_i % (PQ);

    int p = npq_residual / Q;
    int q = npq_residual % Q;

    Accumulator accum = 0;

    for (int gemm_k = 0; gemm_k < GEMM_K; ++gemm_k) {

      int k = gemm_j;

      int c = gemm_k / (RS);
      int crs_residual = gemm_k % (RS);

      int r = crs_residual / S;
      int s = crs_residual % S;

      int h = f(p, r);
      int w = g(q, s);

      ElementA a = tensor_A.at({n, h, w, c});
      ElementB b = tensor_B.at({k, r, s, c});

      accum += a * b;
    }

    C[gemm_i * K + gemm_j] = accum;
  }
}
```

The [CUTLASS GEMM implementation](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html) explicitly iterates over tiles. Consequently,
a tile iterator could be implemented to compute these functions analytically and load the appropriate
elements. However, the resulting modulo arithmetic would be computationally intensive, and overhead would
limit performance of a GEMM kernel targeting Turing Tensor Cores.

The following section describes how an efficient implementation may be implemented within the structure of
a hierarchical GEMM kernel targeting Tensor Cores.
