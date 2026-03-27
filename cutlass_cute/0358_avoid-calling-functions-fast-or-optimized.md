---
title: "Avoid calling functions “fast” or “optimized”"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/programming_guidelines.html#avoid-calling-functions-fast-or-optimized"
---

#### [Avoid calling functions “fast” or “optimized”](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#avoid-calling-functions-fast-or-optimized)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#avoid-calling-functions-fast-or-optimized "Permalink to this headline")

Putting words like “fast” or “optimized”
in the name of a function
assumes that the “fast” path is actually faster.
That might be true now, but later changes
(in the code, compilers, or GPU hardware)
might make it false.  In that case,
your name could be unintentionally misleading.
Consider instead a name that briefly describes
the algorithm or feature that is relevant for optimization.
For example, `compute_on_host` is more meaningful
than `compute_slowly`, and computing on host
might be faster in some cases
(e.g., if the data are already on host
and the algorithm is not GPU-friendly).

CUTLASS code has not always followed this rule in the past.
Some functions and classes might have words like “fast” in their name.
New code should follow this rule, however.
