---
title: "5.5.1.2. Normal and Subnormal Values"
section: "5.5.1.2"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/mathematical-functions.html#normal-and-subnormal-values"
---

### [5.5.1.2. Normal and Subnormal Values](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#normal-and-subnormal-values)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#normal-and-subnormal-values "Permalink to this headline")

Any floating-point value with an exponent field that is neither all zeros nor all ones is called _normal_.

An important aspect of floating-point values is the wide gap between the smallest representable positive normal number, `FLT_MIN`, and zero. This gap is much wider than the gap between `FLT_MIN` and the second-smallest normal number.

Floating-point _subnormal_ numbers, also called _denormals_, were introduced to address this issue. A subnormal floating-point value is represented with all bits in the exponent set to zero and at least one bit set in the significand. Subnormals are a required part of the IEEE-754 floating-point standard.

Subnormal numbers allow for a gradual loss of precision as an alternative to sudden rounding toward zero. However, subnormal numbers are computationally more expensive. Therefore, applications that don’t require strict accuracy may choose to avoid them to improve performance. The `nvcc` compiler allows disabling subnormal numbers by setting the `-ftz=true` option (flush-to-zero), which is also included in `--use_fast_math`.

A simplified visualization of the encoding of the smallest normal value and subnormal values in single-precision is shown in the following figure:

![minimum normal value and subnormal values representations](images/______-___-_________-_______1.png)

where `X` represents both `0` and `1`.
