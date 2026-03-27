---
title: "5.5.1.3. Special Values"
section: "5.5.1.3"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/mathematical-functions.html#special-values"
---

### [5.5.1.3. Special Values](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#special-values)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#special-values "Permalink to this headline")

The IEEE-754 standard defines three special values for floating-point numbers:

****Zero:****

- Mathematical zero.
- Note that there are two possible representations of floating-point zero: `+0` and `-0`. This differs from the representation of integer zero.
- `+0 == -0` evaluates to `true`.
- Zero is encoded with all bits set to `0` in the exponent and significand.

****Infinity:****

- Floating-point numbers behave according to saturation arithmetic, in which operations that overflow the representable range result in `+Infinity` or `-Infinity`.
- Infinity is encoded with all bits in the exponent set to `1` and all bits in the significand set to `0`. There are exactly two encodings for infinity values.
- Arithmetic operations involving infinity and finite nonzero values typically result in infinity. Indeterminate forms such as `Inf * 0.0`, `Inf - Inf`, `Inf / Inf`, and `0.0 / 0.0` result in NaN.

****Not-a-Number (NaN):****

- NaN is a special symbol that represents an undefined or non-representable value. Common examples are `0.0 / 0.0`,  `sqrt(-1.0)`, or `+Inf - Inf`.
- NaN is encoded with all bits in the exponent set to `1` and any bit pattern in the significand, except for all bits set to 0. There are \(\(2^{\mathrm{mantissa} + 1} - 2\)\) possible encodings.
- Any arithmetic operation involving a NaN will result in NaN.
- Any ordered comparison (`<`, `<=`, `>`, `>=`, `==`) involving a NaN will result in `false`, including `NaN == NaN` (non-reflexive). The unordered comparison `NaN != NaN` returns `true`.
- NaNs are provided in two forms:
  - Quiet NaNs `qNaN` are used to propagate errors resulting from invalid operations or values. Invalid arithmetic operations generally produce a quiet NaN. They are encoded with the most significant bit of the significand set to `1`.
  - Signaling NaNs `sNaN` are designed to raise an invalid-operation exception. Signaling NaNs are generally explicitly created. They are encoded with the most significant bit of the significand set to `0`.
  - The exact bit patterns for Quiet and Signaling NaNs are implementation-defined. CUDA provides the [cuda::std::numeric_limits<T>::quiet_NaN](https://en.cppreference.com/w/cpp/types/numeric_limits/quiet_NaN.html) and [cuda::std::numeric_limits<T>::signaling_NaN](https://en.cppreference.com/w/cpp/types/numeric_limits/signaling_NaN.html) constants to get their special values.

A simplified visualization of the encodings of special values is shown in the following figure:

![Floating-Point Representation for Infinity and NaN](images/_______-_______1.png)

where `X` represents both `0` and `1`.
