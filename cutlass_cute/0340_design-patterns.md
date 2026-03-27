---
title: "Design Patterns"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/programming_guidelines.html#design-patterns"
---

## [Design Patterns](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#design-patterns)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#design-patterns "Permalink to this headline")

CUTLASS aims for the highest performance possible on NVIDIA GPUs.
It also offers flexible components that can be assembled and customized
to solve new problems related to deep learning and linear algebra.
Given a tradeoff between simplicity and performance,
CUTLASS chooses performance.
Consequently, several design patterns are necessary
to yield a composable structure
while also satisfying these performance objectives.
