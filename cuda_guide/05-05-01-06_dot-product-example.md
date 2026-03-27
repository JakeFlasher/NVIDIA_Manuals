---
title: "5.5.1.6. Dot Product Example"
section: "5.5.1.6"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/mathematical-functions.html#dot-product-example"
---

### [5.5.1.6. Dot Product Example](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#dot-product-example)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#dot-product-example "Permalink to this headline")

Consider the problem of finding the dot product of two short vectors \(\(\overrightarrow{a}\)\) and \(\(\overrightarrow{b}\)\) both with four elements.

$$
\[\begin{split}\overrightarrow{a} = \begin{bmatrix} a_{1} \\ a_{2} \\ a_{3} \\ a_{4} \end{bmatrix}
\qquad
\overrightarrow{b} = \begin{bmatrix} b_{1} \\ b_{2} \\ b_{3} \\ b_{4} \end{bmatrix}
\qquad
\overrightarrow{a} \cdot \overrightarrow{b} = a_{1}b_{1} + a_{2}b_{2} + a_{3}b_{3} + a_{4}b_{4}\end{split}\]
$$

Although this operation is easy to write down mathematically, implementing it in software involves several alternatives that could lead to slightly different results. All of the strategies presented here use operations that are fully compliant with IEEE-754.

**Example Algorithm 1:** The simplest way to compute the dot product is to use a sequential sum of products, keeping the multiplications and additions separate.

> The final result can be represented as \(\(((((a_1 \times b_1) + (a_2 \times b_2)) + (a_3 \times b_3)) + (a_4 \times b_4))\)\).

**Example Algorithm 2:** Compute the dot product sequentially using fused multiply-add.

> The final result can be represented as \(\((a_4 \times b_4) + ((a_3 \times b_3) + ((a_2 \times b_2) + (a_1 \times b_1 + 0)))\)\).

**Example Algorithm 3:** Compute the dot product using a divide-and-conquer strategy. First, we find the dot products of the first and second halves of the vectors. Then, we combine these results using addition. This algorithm is called the “parallel algorithm” because the two subproblems can be computed in parallel since they are independent of each other. However, the algorithm does not require a parallel implementation; it can be implemented with a single thread.

> The final result can be represented as \(\(((a_1 \times b_1) + (a_2 \times b_2)) + ((a_3 \times b_3) + (a_4 \times b_4))\)\).
