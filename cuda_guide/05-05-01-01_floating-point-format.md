---
title: "5.5.1.1. Floating-Point Format"
section: "5.5.1.1"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/mathematical-functions.html#floating-point-format"
---

### [5.5.1.1. Floating-Point Format](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#floating-point-format)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#floating-point-format "Permalink to this headline")

Floating-point format and functionality are defined in the [IEEE-754 Standard](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8766229).

The standard mandates that binary floating-point data be encoded on three fields:

- **Sign**: one bit to indicate a positive or negative number.
- **Exponent**: encodes the base 2 exponent offset by a numeric bias.
- **Significand** (also called _mantissa_ or _fraction_): encodes the fractional value of the number.

![Floating-Point Encoding](images/________-_____-_______1.png)

The latest IEEE-754 standard defines the encodings and properties of the following binary formats:

- 16-bit, also known as half-precision, corresponding to the `__half` data type in CUDA.
- 32-bit, also known as single-precision, corresponding to the `float` data type in C, C++, and CUDA.
- 64-bit, also known as double-precision, corresponding to the `double` data type in C, C++, and CUDA.
- 128-bit, also known as quad-precision, corresponding to the `__float128` or `_Float128` data types in CUDA.

These types have the following bit lengths:

![IEEE-754 Floating-Point Encodings](images/________-_____-_______2.png)

The numeric value associated with floating-point encoding for [normal](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#normal-subnormal) values is computed as follows:

$$
\[(-1)^\mathrm{sign} \times 1.\mathrm{mantissa} \times 2^{\mathrm{exponent} - \mathrm{bias}}\]
$$

For [subnormal](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#normal-subnormal) values, the formula is modified to:

$$
\[(-1)^\mathrm{sign} \times 0.\mathrm{mantissa} \times 2^{1-\mathrm{bias}}\]
$$

| The exponents are biased by \(\(127\)\) and \(\(1023\)\) for single- and double-precision, respectively. The integral part of \(\(1.\)\) is implicit in the fraction.
| For example, the value \(\(-192 = (-1)^1 \times 2^7 \times 1.5\)\), and is encoded as a negative sign, an exponent of \(\(7\)\), and a fractional part \(\(0.5\)\). Hence the exponent \(\(7\)\) is represented by bit strings with values `7 + 127 = 134 = 10000110` for `float` and `7 + 1023 = 1030 = 10000000110` for `double`. The mantissa `0.5 = 2^-1` is represented by a binary value with `1` in the first position. The binary encodings of \(\(-192\)\) in single-precision and double-precision are shown in the following figure:

![Floating-Point Representation for ``-192``](images/________-_____-_______3.png)

Since the fraction field uses a limited number of bits, not all real numbers can be represented exactly. For instance, the binary representation of the mathematical value of the fraction \(\(2 / 3\)\) is `0.10101010...`, which has an infinite number of bits after the binary point. Therefore, \(\(2 / 3\)\) must be rounded before it can be represented as a floating-point number with limited precision. The rounding rules and modes are specified in IEEE-754. The most frequently used mode is _round-to-nearest-ties-to-even_, abbreviated round-to-nearest.
