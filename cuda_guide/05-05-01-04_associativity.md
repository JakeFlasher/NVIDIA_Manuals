---
title: "5.5.1.4. Associativity"
section: "5.5.1.4"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/mathematical-functions.html#associativity"
---

### [5.5.1.4. Associativity](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#associativity)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#associativity "Permalink to this headline")

It is important to note that the rules and properties of mathematical arithmetic do not directly apply to floating-point arithmetic due to its limited precision. The example below shows single-precision values `A`, `B`, and `C` and the exact mathematical value of their sum computed using different associativity.

$$
\[\begin{split}\begin{aligned}
A           &= 2^{1} \times 1.00000000000000000000001 \\
B           &= 2^{0} \times 1.00000000000000000000001 \\
C           &= 2^{3} \times 1.00000000000000000000001 \\
(A + B) + C &= 2^{3} \times 1.01100000000000000000001011 \\
A + (B + C) &= 2^{3} \times 1.01100000000000000000001011
\end{aligned}\end{split}\]
$$

Mathematically, \(\((A + B) + C\)\) is equal to \(\(A + (B + C)\)\).

Let \(\(\mathrm{rn}(x)\)\) denote one rounding step on \(\(x\)\). Performing the same computations in single-precision floating-point arithmetic in round-to-nearest mode according to IEEE-754, we obtain:

$$
\[\begin{split}\begin{aligned}
A + B                                     &= 2^{1} \times 1.1000000000000000000000110000\ldots \\
\mathrm{rn}(A+B)                          &= 2^{1} \times 1.10000000000000000000010 \\
B + C                                     &= 2^{3} \times 1.0010000000000000000000100100\ldots \\
\mathrm{rn}(B+C)                          &= 2^{3} \times 1.00100000000000000000001 \\
A + B + C                                 &= 2^{3} \times 1.0110000000000000000000101100\ldots \\
\mathrm{rn}\big(\mathrm{rn}(A+B) + C\big) &= 2^{3} \times 1.01100000000000000000010 \\
\mathrm{rn}\big(A + \mathrm{rn}(B+C)\big) &= 2^{3} \times 1.01100000000000000000001
\end{aligned}\end{split}\]
$$

For reference, the exact mathematical results are also computed above. The results computed according to IEEE-754 differ from the exact mathematical results. Additionally, the results corresponding to the sums  \(\(\mathrm{rn}(\mathrm{rn}(A + B) + C)\)\) and  \(\(\mathrm{rn}(A + \mathrm{rn}(B + C))\)\) differ from each other. In this case, \(\(\mathrm{rn}(A + \mathrm{rn}(B + C))\)\) is closer to the correct mathematical result than \(\(\mathrm{rn}(\mathrm{rn}(A + B) + C)\)\).

This example shows that seemingly identical computations can produce different results, even when all basic operations comply with IEEE-754.
