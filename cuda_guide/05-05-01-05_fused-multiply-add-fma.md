---
title: "5.5.1.5. Fused Multiply-Add (FMA)"
section: "5.5.1.5"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/mathematical-functions.html#fused-multiply-add-fma"
---

### [5.5.1.5. Fused Multiply-Add (FMA)](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#fused-multiply-add-fma)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#fused-multiply-add-fma "Permalink to this headline")

The Fused Multiply-Add (FMA) operation computes the result with only one rounding step. Without the FMA, the result would require two rounding steps: one for multiplication and one for addition. Because the FMA uses only one rounding step, it produces a more accurate result.

The Fused Multiply-Add operation can affect the propagation of NaNs differently than two separate operations. However, FMA NaN handling is not universally identical across all targets. Different implementations with multiple NaN operands may prefer a quiet NaN or propagate one operand’s payload. Additionally, IEEE-754 does not strictly mandate a deterministic payload selection order when multiple NaN operands are present. NaNs may also occur in intermediate computations, for example, \(\(\infty \times 0 + 1\)\) or \(\(1 \times \infty - \infty\)\), resulting in an implementation-defined NaN payload.

---

For clarity, first consider an example using decimal arithmetic to illustrate how the FMA operation works. We will compute \(\(x^2 - 1\)\) using five total digits of precision, with four digits after the decimal point.

- For \(\(x = 1.0008\)\), the correct mathematical result is \(\(x^2 - 1 = 1.60064 \times 10^{-4}\)\). The closest number using only four digits after the decimal point is \(\(1.6006 \times 10^{-4}\)\).
- The Fused Multiply-Add operation achieves the correct result using only one rounding step \(\(\mathrm{rn}(x \times x - 1) = 1.6006 \times 10^{-4}\)\).
- The alternative is to compute the multiply and add steps separately. \(\(x^2 = 1.00160064\)\) translates to \(\(\mathrm{rn}(x \times x) = 1.0016\)\). The final result is \(\(\mathrm{rn}(\mathrm{rn}(x \times x) -1) = 1.6000 \times 10^{-4}\)\).

Rounding the multiply and add separately yields a result that is off by \(\(0.00064\)\). The corresponding FMA computation is wrong by only \(\(0.00004\)\) and its result is closest to the correct mathematical answer. The results are summarized below:

$$
\[\begin{split}\begin{aligned}
x                                           &= 1.0008 \\
x^{2}                                       &= 1.00160064 \\
x^{2} - 1                                   &= 1.60064 \times 10^{-4} && \text{true value} \\
\mathrm{rn}\big(x^{2} - 1\big)              &= 1.6006 \times 10^{-4} && \text{fused multiply-add} \\
\mathrm{rn}\big(x^{2}\big)                  &= 1.0016 \\
\mathrm{rn}\big(\mathrm{rn}(x^{2}) - 1\big) &= 1.6000 \times 10^{-4} && \text{multiply, then add}
\end{aligned}\end{split}\]
$$

---

Below is another example, using binary single precision values:

$$
\[\begin{split}\begin{aligned}
A                                                &= 2^{0} \times 1.00000000000000000000001 \\
B                                                &= -2^{0} \times 1.00000000000000000000010 \\
\mathrm{rn}\big(A \times A + B\big)              &= 2^{-46} \times 1.00000000000000000000000 && \text{fused multiply-add} \\
\mathrm{rn}\big(\mathrm{rn}(A \times A) + B\big) &= 0 && \text{multiply, then add}
\end{aligned}\end{split}\]
$$

- Computing multiplication and addition separately results in the loss of all bits of precision, yielding \(\(0\)\).
- Computing the FMA, on the other hand, provides a result equal to the mathematical value.

Fused multiply-add helps prevent loss of precision during subtractive cancellation. Subtractive cancellation occurs when quantities of similar magnitude with opposite signs are added. In this case, many of the leading bits cancel out, resulting in fewer meaningful bits. The fused multiply-add computes a double-width product during multiplication. Thus, even if subtractive cancellation occurs during addition, there are enough valid bits remaining in the product to yield a precise result.

---

**Fused Multiply-Add Support in CUDA:**

CUDA provides the Fused Multiply-Add operation in several ways for both `float` and `double` data types:

- `x * y + z` when compiled with the flags `-fmad=true` or `--use_fast_math`.
- `fma(x, y, z)` and `fmaf(x, y, z)` [C Standard Library functions](https://en.cppreference.com/w/c/numeric/math/fma).
- `__fmaf_[rd, rn, ru, rz]`, `__fmaf_ieee_[rd, rn, ru, rz]`, and `__fma_[rd, rn, ru, rz]` [CUDA mathematical intrinsic functions](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html).
- `cuda::std::fma(x, y, z)` and `cuda::std::fmaf(x, y, z)` [CUDA C++ Standard Library functions](https://en.cppreference.com/w/cpp/numeric/math/fma.html).

---

**Fused Multiply-Add Support on Host Platforms:**

Whether to use the fused operation depends on the availability of the operation on the platform and how the code is compiled. It is important to understand the host platform’s support for Fused Multiply-Add when comparing CPU and GPU results.

- Compiler flags and Fused Multiply-Add hardware support:
  - `-mfma` with [GCC](https://gcc.gnu.org/onlinedocs/gcc/x86-Options.html#index-mmmx) and [Clang](https://clang.llvm.org/docs/UsersManual.html#cmdoption-ffp-contract), `-Mfma` with [NVC++](https://docs.nvidia.com/hpc-sdk/compilers/hpc-compilers-user-guide/index.html#gpu), and `/fp:contract` with [Microsoft Visual Studio](https://learn.microsoft.com/en-us/cpp/preprocessor/fp-contract).
  - x86 platforms with the AVX2 ISA, for example, code compiled with the `-mavx2` flag using GCC or Clang, and `/arch:AVX2` with Microsoft Visual Studio.
  - Arm64 (AArch64) platforms with Advanced SIMD (Neon) ISA.
- `fma(x, y, z)` and `fmaf(x, y, z)` [C Standard Library functions](https://en.cppreference.com/w/c/numeric/math/fma).
- `std::fma(x, y, z)` and `std::fmaf(x, y, z)` [C++ Standard Library functions](https://en.cppreference.com/w/cpp/numeric/math/fma.html).
- `cuda::std::fma(x, y, z)` and `cuda::std::fmaf(x, y, z)` [CUDA C++ Standard Library functions](https://en.cppreference.com/w/cpp/numeric/math/fma.html).
