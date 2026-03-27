---
title: "Fundamental Types"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/fundamental_types.html#fundamental-types"
---

# [Fundamental Types](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#fundamental-types)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#fundamental-types "Permalink to this headline")

CUTLASS defies several fundamental numeric and container classes upon which computations and
algorithms algorithms for linear algebra computations are implemented.

Where possible, CUTLASS fundamental types mirror the C++ Standard Library. However, there are circumstances that necessitate divergence from the Standard Library’s specification. In such cases, the CUTLASS implementation adopts unique capitalization to distinguish that standard vocabulary types may not be safely substituted in all cases.

Most types in CUTLASS are usable in both host code and device code. Moreover, they are functional regardless of compute capability, but they may only be efficient when hardware support is present.
