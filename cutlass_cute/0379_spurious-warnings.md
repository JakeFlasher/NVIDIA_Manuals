---
title: "Spurious warnings"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/programming_guidelines.html#spurious-warnings"
---

#### [Spurious warnings](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#spurious-warnings)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#spurious-warnings "Permalink to this headline")

Some compilers, or some versions of a compiler, emit spurious warnings, that is, “false positives” for perfectly fine code.  While such code is correct, the warnings can obscure errors.  Users also may report warnings as bugs, and processing those bugs takes developer time away from other tasks.  Thus, it’s good to try to “fix” the warnings, if doing so wouldn’t make the code worse.
