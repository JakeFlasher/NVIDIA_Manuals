---
title: "Example 1 – Worked Example of Calculating a Composition"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/02_layout_algebra.html#example-1-worked-example-of-calculating-a-composition"
---

#### [Example 1 – Worked Example of Calculating a Composition](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute#example-1-worked-example-of-calculating-a-composition)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/#example-1-worked-example-of-calculating-a-composition "Permalink to this headline")

We provide a more complex example of composition, where both operand layouts are multi-modal to illustrate the concepts introduced above.

```console
Functional composition, R := A o B
R(c) := (A o B)(c) := A(B(c))

Example
A = (6,2):(8,2)
B = (4,3):(3,1)

1. Using the left-distributive and concatenation properties for layouts we write the composition as,

R = A o B
  = (6,2):(8,2) o (4,3):(3,1)
  = ((6,2):(8,2) o 4:3, (6,2):(8,2) o 3:1)

---
1. Compute `(6,2):(8,2) o 4:3`

- First, we compute the strided layout,

(6,2):(8,2) / 3 = (6/3,2):(8*3,2) = (2,2):(24,2)

- Next, we keep the shape compatible,

(2,2):(24,2) % 4 = (2,2):(24,2)

---
2. Compute `(6,2):(8,2) o 3:1`

- First, we compute the strided layout

(6,2):(8,2) / 1 = (6,2):(8,2)

- Next, we keep the shape compatible,

(6,2):(8,2) % 3 = (3,1):(8,2)

---

Putting this together and coalescing each mode, we obtain the result

R = A o B
  = ((2, 2), 3): ((24, 2), 8)
```
